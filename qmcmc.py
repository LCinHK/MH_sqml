import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import cirq
import sympy
import collections


# Save kernel outputs via namedtuple (here we only keep target_log_prob)
RWResult = collections.namedtuple("RWResult", "target_log_prob")


def _to_spin_batch(spins):
    """Ensure `spins` is float32 with shape [batch, n].

    - If input is [n], expand to [1, n]
    - Downstream energy/log-prob computations consistently use a batch dimension
    """
    spins = tf.convert_to_tensor(spins)
    spins = tf.cast(spins, tf.float32)
    if spins.shape.rank == 1:
        spins = tf.expand_dims(spins, axis=0)
    return spins


def ising_log_prob_dense(spins, J, h, temperature):
    """
    Compute the (unnormalized) log_prob for a dense Ising model:
        log p(s) ∝ (-E(s)) / T

    Args:
      spins: [batch, n], values should be in {-1, +1}
      J: [n, n] symmetric coupling matrix (diagonal usually 0)
      h: [n] external field
      temperature: scalar T

    Energy definition:
      E(s) = -1/2 s^T J s - h^T s
    Therefore:
      -E(s) =  1/2 s^T J s + h^T s
      log_prob ∝ (-E)/T
    """
    spins = _to_spin_batch(spins)
    J = tf.cast(J, tf.float32)
    h = tf.cast(h, tf.float32)
    temperature = tf.cast(temperature, tf.float32)

    # pair = s^T J s (batched) via einsum
    pair = tf.einsum("bi,ij,bj->b", spins, J, spins)
    # field = h^T s
    field = tf.einsum("bi,i->b", spins, h)
    # -E(s) = 0.5 * s^T J s + h^T s
    neg_E = 0.5 * pair + field
    return neg_E / temperature


def upper_triangle_edges(n):
    """Generate a full upper-triangular edge list (i, j), 0 <= i < j < n, in a fixed order.

    Note: the current implementation mainly uses `edges_from_J` to generate a sparse edge set.
    This helper can be used for fully-connected edges.
    """
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges


def edges_from_J(J, eps=0.0):
    """
    Build a sparse edge list from coupling matrix J:
    include (i, j) only if i < j and |J_ij| > eps.

    Returns: python list[(i, j), ...]

    Important:
    - This runs on the Python side (when building the kernel/circuit), so J must be a concrete value
      (eager tensor / numpy array), not a placeholder or dynamically-changing tensor.
    """
    import numpy as np
    if hasattr(J, "numpy"):
        Jnp = J.numpy()
    else:
        Jnp = np.asarray(J)
    n = Jnp.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Jnp[i, j]) > eps:
                edges.append((i, j))
    return edges


class QuantumMCMCIsingKernel(tfp.python.mcmc.kernel.TransitionKernel):
    """
    Quantum-enhanced MCMC proposal kernel (TransitionKernel).

    - Target: dense Ising (arbitrary symmetric J, h)
    - Proposal: use a TFQ sampling layer to produce the next candidate state (bits),
      from a "current-state encoding + trotterized parameterized circuit", then map back to spins.

    Entangling operations are applied only on the sparse edge set
        E = {(i, j): |J_ij| > eps}
    so the circuit depth/parameter count scales with |E| (instead of n^2).
    """

    def __init__(self, size, J, h, r, temp, eps=0.0):
        # Number of spins n
        self._n = int(size)

        # Unify dtype
        J = tf.cast(J, dtype=tf.float32)
        h = tf.cast(h, dtype=tf.float32)
        temp = tf.cast(temp, dtype=tf.float32)

        # TFP kernels typically store config in _parameters (for tracing/serialization/property access)
        self._parameters = dict(
            target_log_prob_fn=self.ising_model_log_prob,
            size_q=self._n,
            J=J,
            h=h,
            rep=int(r),             # trotter repetition count
            temperature=temp,
        )

        # TFQ sampling layer: given circuit + params, returns measurement results (0/1 bits)
        self.sample = tfq.layers.Sample()

        # Use a 1D line of GridQubit(0, i) for n qubits
        self.qubits = [cirq.GridQubit(0, i) for i in range(self._n)]

        # Alpha scaling: normalize (J, h) scale into circuit parameters
        # - Avoid tf.norm(J, ord="fro") because some TF versions don't support it
        J_norm = tf.sqrt(tf.reduce_sum(tf.square(J)))   # Frobenius norm
        h_norm = tf.norm(h)                             # L2 norm
        denom = tf.sqrt(tf.square(J_norm) + tf.square(h_norm) + 1e-12)
        denom = tf.maximum(denom, 1e-6)                 # prevent divide-by-zero / tiny denom blow-up
        self.alpha = tf.sqrt(tf.cast(self._n, tf.float32)) / denom

        # --- Sparse edges: fixed at construction time (determined on Python side) ---
        # Note: eps controls sparsity; larger eps -> fewer edges -> sparser circuit.
        self._edges = edges_from_J(J, eps=eps)
        self._num_edges = len(self._edges)

        # --- Symbolic params for circuit (sympy symbols) ---
        # xs: encodes the "current bitstring" into the circuit (X**xs[i])
        # a, b: single-qubit rotation params (shared a + per-qubit b_i)
        # theta: one entangling param per sparse edge
        a = sympy.Symbol("a")
        b = sympy.symbols(f"b0:{self._n}")                      # n symbols
        theta = sympy.symbols(f"theta0:{self._num_edges}")      # |E| symbols (can be 0)
        xs = sympy.symbols(f"xs0:{self._n}")                    # n symbols

        # Build parameterized circuit (depends on xs, a, b, theta)
        self.trotterized_circuit = self.make_circuit(self.rep, xs, a, b, theta)

        # TFQ requires symbol_names and symbol_values to match in length and order
        # We flatten all symbols into a fixed ordering
        self.params = list(xs) + [a] + list(b) + list(theta)

        # Cache total symbol count: n(xs) + 1(a) + n(b) + |E|(theta) = 2n + 1 + |E|
        self._num_symbols = 2 * self._n + 1 + self._num_edges

    # Typical TFP-kernel properties: expose read-only config from _parameters
    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def size_q(self):
        return self._parameters["size_q"]

    @property
    def J(self):
        return self._parameters["J"]

    @property
    def h(self):
        return self._parameters["h"]

    @property
    def rep(self):
        return self._parameters["rep"]

    @property
    def temperature(self):
        return self._parameters["temperature"]

    @property
    def is_calibrated(self):
        # This is an "uncalibrated" proposal kernel, so typically False
        return False

    def ising_model_log_prob(self, spins):
        """Target distribution log_prob (unnormalized)."""
        return ising_log_prob_dense(spins, self.J, self.h, self.temperature)

    def make_circuit(self, r, xs, a, b, theta):
        """
        Build the trotterized quantum proposal circuit.

        High-level structure:
          1) Encode current bitstring via X**xs[i] (xs ∈ {0, 1})
          2) Repeat r times (trotter steps):
             - For each qubit: RX(2a), RZ(2b_i)
             - For each sparse edge (i, j): implement a ZZ-phase coupling using CNOT-RZ(theta_e)-CNOT
          3) Final layer of single-qubit RX/RZ

        Notes:
          - CNOT-RZ-CNOT is a common decomposition for exp(-i * theta/2 * Z⊗Z) (convention-dependent)
          - Entangling is applied only on self._edges to reduce parameters and gate count
        """
        circuit = cirq.Circuit()

        # (1) Encode initial bitstring:
        # If xs[i]=1 apply X on qubit i; if xs[i]=0 then X**0 = I
        for i, q in enumerate(self.qubits):
            circuit += cirq.X(q) ** xs[i]

        # (2) Trotter repetition layers
        for _ in range(r):
            # Single-qubit rotations
            for i, q in enumerate(self.qubits):
                circuit += cirq.rx(2 * a).on(q)
                circuit += cirq.rz(2 * b[i]).on(q)

            # Sparse-edge entangling: one theta[e_idx] per edge
            for e_idx, (i, j) in enumerate(self._edges):
                qi, qj = self.qubits[i], self.qubits[j]
                circuit += cirq.CNOT(qi, qj)
                circuit += cirq.rz(theta[e_idx]).on(qj)
                circuit += cirq.CNOT(qi, qj)

        # (3) Final single-qubit rotation layer
        for i, q in enumerate(self.qubits):
            circuit += cirq.rx(2 * a).on(q)
            circuit += cirq.rz(2 * b[i]).on(q)

        return circuit

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """
        Perform one MCMC "proposal generation" step.

        Note: the outer MetropolisHastings wrapper will compute acceptance/rejection using target_log_prob.
        This method only generates the candidate next_state and updates kernel results accordingly.
        """
        # Randomize tuning: gamma controls mixing; t controls "evolution time"; dt = t/rep
        gamma = tf.random.uniform(shape=[], minval=0.25, maxval=0.6, seed=seed)
        t = tf.random.uniform(shape=[], minval=2.0, maxval=20.0, seed=seed)
        dt = t / tf.cast(self.rep, tf.float32)

        # a is a global RX parameter (scalar)
        a = gamma * dt

        # b is per-qubit RZ parameter vector [n]
        # Depends on external field h and includes scaling by alpha, dt, and (1-gamma)
        b = -(1.0 - gamma) * self.alpha * self.h * dt
        b = tf.cast(tf.reshape(b, [-1]), tf.float32)  # force rank-1: [n]

        # theta is the entangling parameter vector [E] for sparse edges
        # Must match the edge ordering used when building the circuit (self._edges)
        if self._num_edges > 0:
            # ij: constant index table (i, j) with shape [E, 2]
            ij = tf.constant(self._edges, dtype=tf.int32)
            # Gather Jij from dense J at those edge indices -> [E]
            Jij = tf.gather_nd(self.J, ij)
            # Compute per-edge theta
            theta = -2.0 * Jij * (1.0 - gamma) * self.alpha * dt
            theta = tf.cast(tf.reshape(theta, [-1]), tf.float32)  # [E]
        else:
            # If there are 0 edges, theta must be a length-0 vector to avoid TFQ mismatch
            theta = tf.zeros([0], dtype=tf.float32)

        # Encode current spins into bits for X**xs[i]:
        # spins: +1 -> 0, -1 -> 1
        # Formula: x_bits = (s - 1) / (-2)
        x_bits = (current_state[0] - 1.0) / (-2.0)
        x_bits = tf.cast(tf.reshape(x_bits, [-1]), tf.float32)  # [n]

        # a1: reshape scalar a to rank-1 [1] for concatenation
        a1 = tf.reshape(tf.cast(a, tf.float32), [1])  # [1]

        # Build TFQ symbol_values in the exact same order as self.params: xs, a, b, theta
        values = tf.concat([x_bits, a1, b, theta], axis=0)  # [2n + 1 + E]

        # TFQ requirement: len(symbol_names) == last-dim length of symbol_values
        tf.debugging.assert_equal(
            tf.shape(values)[0],
            self._num_symbols,
            message="TFQ symbol_values length != symbol_names length"
        )

        # Call TFQ Sample: input circuit + parameters; output is measured bits 0/1
        # symbol_values must be [batch, num_symbols], so expand to [1, ...]
        next_bits = self.sample(
            self.trotterized_circuit,
            symbol_names=self.params,
            symbol_values=tf.expand_dims(values, axis=0),   # [1, num_symbols]
            repetitions=1,
        ).to_tensor()[0]  # take batch=0 -> [n] bits

        # bits -> spins: 0 -> +1, 1 -> -1
        next_spins = tf.cast(next_bits * -2 + 1, tf.float32)
        next_spins = tf.reshape(next_spins, [1, self.size_q])  # [1, n]

        # Compute target_log_prob for the candidate state (used by outer MH)
        next_target_log_prob = self.target_log_prob_fn(next_spins)

        # Update kernel_results (keep structure; replace target_log_prob)
        new_kernel_results = previous_kernel_results._replace(target_log_prob=next_target_log_prob)
        return next_spins, new_kernel_results

    def bootstrap_results(self, init_state):
        """Initialize kernel_results by storing the initial state's target_log_prob."""
        return RWResult(target_log_prob=self.target_log_prob_fn(init_state))


class ClassicalMCMCIsingKernel(tfp.python.mcmc.kernel.TransitionKernel):
    """
    Classical proposal kernel (global independent proposal):
    - Each step proposes a uniformly random state from {-1, +1}^n
    - Target remains the dense Ising (J, h)
    """

    def __init__(self, size, J, h, r, temp):
        # r is unused here, but kept to match the interface
        self._parameters = dict(
            target_log_prob_fn=self.ising_model_log_prob,
            size_q=int(size),
            J=tf.cast(J, tf.float32),
            h=tf.cast(h, tf.float32),
            temperature=tf.cast(temp, tf.float32),
        )

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def size_q(self):
        return self._parameters["size_q"]

    @property
    def J(self):
        return self._parameters["J"]

    @property
    def h(self):
        return self._parameters["h"]

    @property
    def temperature(self):
        return self._parameters["temperature"]

    @property
    def is_calibrated(self):
        return False

    def ising_model_log_prob(self, spins):
        return ising_log_prob_dense(spins, self.J, self.h, self.temperature)

    def one_step(self, current_state, previous_kernel_results, seed=None):
        # Uniformly sample bits ∈ {0, 1}, then map to spins ∈ {-1, +1}
        next_spins = tf.cast(
            tf.random.uniform(
                shape=[1, self.size_q], minval=0, maxval=2, dtype=tf.int32, seed=seed
            )
            * 2
            - 1,
            tf.float32,
        )
        next_target_log_prob = self.target_log_prob_fn(next_spins)
        new_kernel_results = previous_kernel_results._replace(
            target_log_prob=tf.cast(next_target_log_prob, tf.float32)
        )
        return next_spins, new_kernel_results

    def bootstrap_results(self, init_state):
        return RWResult(target_log_prob=self.target_log_prob_fn(init_state))


class ClassicalMCMCIsingKernel_local(tfp.python.mcmc.kernel.TransitionKernel):
    """
    Classical local-move proposal kernel:
    - Each step randomly selects one position and flips that spin (single-spin flip)
    - Target: dense Ising (J, h)
    """

    def __init__(self, size, J, h, r, temp):
        self._parameters = dict(
            target_log_prob_fn=self.ising_model_log_prob,
            size_q=int(size),
            J=tf.cast(J, tf.float32),
            h=tf.cast(h, tf.float32),
            temperature=tf.cast(temp, tf.float32),
        )

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def size_q(self):
        return self._parameters["size_q"]

    @property
    def J(self):
        return self._parameters["J"]

    @property
    def h(self):
        return self._parameters["h"]

    @property
    def temperature(self):
        return self._parameters["temperature"]

    @property
    def is_calibrated(self):
        return False

    def ising_model_log_prob(self, spins):
        return ising_log_prob_dense(spins, self.J, self.h, self.temperature)

    def one_step(self, current_state, previous_kernel_results, seed=None):
        # Note: under tf.function, set_seed semantics may differ from eager;
        # this is only a best-effort attempt for reproducibility.
        if seed is not None:
            tf.random.set_seed(seed)

        # Ensure shape [batch, n]; typically [1, n] here
        x = _to_spin_batch(current_state)
        n = tf.shape(x)[1]

        # Randomly choose an index to flip
        idx = tf.random.uniform([], minval=0, maxval=n, dtype=tf.int32)

        # mask is one-hot: selected position is 1, others 0
        mask = tf.one_hot(idx, depth=n, dtype=tf.float32)  # [n]
        mask = tf.reshape(mask, [1, n])                    # [1, n]

        # Flip: s -> -s
        # x * (1 - 2*mask):
        # - where mask=1: multiply by (1-2)=-1 to flip
        # - where mask=0: multiply by 1 to keep unchanged
        next_spins = x * (1.0 - 2.0 * mask)

        next_target_log_prob = self.target_log_prob_fn(next_spins)
        new_kernel_results = previous_kernel_results._replace(
            target_log_prob=tf.cast(next_target_log_prob, tf.float32)
        )
        return tf.cast(next_spins, tf.float32), new_kernel_results

    def bootstrap_results(self, init_state):
        return RWResult(target_log_prob=self.target_log_prob_fn(init_state))


class IsingMH(object):
    """
    Metropolis-Hastings wrapper:
    - Selects a proposal kernel (base kernel) based on kernel_type
    - Wraps it with tfp.mcmc.MetropolisHastings for accept/reject

    Inputs:
      - J: [n, n]
      - h: [n]

    kernel_type:
      - "classical": global independent uniform proposal
      - "local": single-spin flip proposal
      - "quantum": TFQ quantum-circuit proposal (sparse entangling edges)
    """

    def __init__(self, size, J, h, r, temp, kernel_type):
        self.n = int(size)
        if kernel_type == "quantum":
            base = QuantumMCMCIsingKernel(size, J, h, r, temp)
        elif kernel_type == "local":
            base = ClassicalMCMCIsingKernel_local(size, J, h, r, temp)
        else:
            base = ClassicalMCMCIsingKernel(size, J, h, r, temp)

        # Wrap with MH: base.one_step provides proposals, MH ensures correct acceptance logic
        self.kernel = tfp.mcmc.MetropolisHastings(base)

    def run_mcmc(self, num_results, num_burnin, init_state=None):
        # If no initial state provided: draw a uniform random state in {-1, +1}^n
        if init_state is None:
            init_state = tf.cast(
                (tf.random.uniform(shape=(1, self.n), minval=0, maxval=2, dtype=tf.int32) * 2) - 1,
                tf.float32,
            )

        @tf.function
        def run_chain():
            # sample_chain returns:
            # - samples: typically [num_results, 1, n]
            # - trace: here trace_fn returns (is_accepted, accepted_results)
            samples, (is_accepted, results) = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin,
                current_state=init_state,
                kernel=self.kernel,
                # pkr is MetropolisHastings kernel_results
                trace_fn=lambda _, pkr: (pkr.is_accepted, pkr.accepted_results),
            )

            # Simple summary stats (reduce_mean/std are over all dimensions)
            sample_mean = tf.reduce_mean(samples)
            sample_stddev = tf.math.reduce_std(samples)
            acc_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            return samples, sample_mean, sample_stddev, acc_rate, results

        return run_chain()