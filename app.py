import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from qmcmc import IsingMH

st.title("Interactive Portfolio Optimization Demo using Quantum-enhanced MCMC")

st.markdown("""
This demo applies quantum-enhanced Markov Chain Monte Carlo (MCMC) to portfolio optimization.
The Ising model represents asset selection: spins (+1/-1) indicate include/exclude assets.
- **J matrix**: Negative correlations penalize selecting correlated assets (risk reduction).
- **h vector**: Expected returns encourage selecting high-return assets.
- Lower energy corresponds to better portfolios.
""")

# Sidebar for inputs
st.sidebar.header("Parameters")

n_assets = st.sidebar.slider("Number of Assets", 4, 20, 8)

st.sidebar.subheader("Expected Returns (h)")
use_random_returns = st.sidebar.checkbox("Use Random Returns", value=True)
if use_random_returns:
    h = np.random.uniform(0.05, 0.15, n_assets)
    st.sidebar.write("Random returns generated.")
else:
    returns_input = st.sidebar.text_area("Enter returns as comma-separated values", "0.1, 0.05, 0.08, 0.12, 0.07, 0.09, 0.06, 0.11")
    try:
        h = np.array([float(x.strip()) for x in returns_input.split(',')])
        if len(h) != n_assets:
            st.error(f"Expected {n_assets} returns, got {len(h)}")
            st.stop()
    except:
        st.error("Invalid returns input")
        st.stop()

st.sidebar.subheader("Correlation Matrix (J)")
use_random_corr = st.sidebar.checkbox("Use Random Correlations", value=True)
if use_random_corr:
    # Generate random correlation matrix
    A = np.random.randn(n_assets, n_assets)
    corr_matrix = np.dot(A, A.T)
    corr_matrix = corr_matrix / np.sqrt(np.diag(corr_matrix)[:, None] * np.diag(corr_matrix)[None, :])
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1)
    st.sidebar.write("Random correlations generated.")
else:
    corr_input = st.sidebar.text_area("Enter correlations as space-separated rows",
                                      "1 0.5 0.3 0.2 0.1 0.4 0.6 0.7\n0.5 1 0.4 0.3 0.2 0.5 0.3 0.4\n0.3 0.4 1 0.5 0.6 0.2 0.1 0.3\n0.2 0.3 0.5 1 0.4 0.3 0.2 0.5\n0.1 0.2 0.6 0.4 1 0.5 0.3 0.4\n0.4 0.5 0.2 0.3 0.5 1 0.6 0.2\n0.6 0.3 0.1 0.2 0.3 0.6 1 0.5\n0.7 0.4 0.3 0.5 0.4 0.2 0.5 1")
    try:
        corr_lines = corr_input.strip().split('\n')
        corr_matrix = np.array([[float(x) for x in line.split()] for line in corr_lines])
        if corr_matrix.shape != (n_assets, n_assets):
            st.error(f"Expected {n_assets}x{n_assets} matrix, got {corr_matrix.shape}")
            st.stop()
    except:
        st.error("Invalid correlation matrix input")
        st.stop()

# Make symmetric and set diagonal to 0 for J
corr_matrix = (corr_matrix + corr_matrix.T) / 2
np.fill_diagonal(corr_matrix, 0)  # No self-correlation
J = -corr_matrix  # Negative for Ising

temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.5, 0.1)
kernel_type = st.sidebar.selectbox("Kernel Type", ["quantum", "classical", "local"])
r = st.sidebar.slider("Trotter Repetitions (r)", 1, 10, 3)
num_steps = st.sidebar.slider("Number of MCMC Steps", 100, 2000, 500)
num_burnin = st.sidebar.slider("Burn-in Steps", 0, 500, 100)

if st.button("Run MCMC"):
    with st.spinner("Running MCMC..."):
        # Convert to tensors
        J_tf = tf.constant(J, dtype=tf.float32)
        h_tf = tf.constant(h, dtype=tf.float32)

        mh = IsingMH(n_assets, J_tf, h_tf, r, temperature, kernel_type)
        samples, sample_mean, sample_stddev, acc_rate, results = mh.run_mcmc(num_steps, num_burnin)

        # Extract energy trace
        target_log_probs = results[0]
        energies = -np.array(target_log_probs) * temperature

        # Final portfolio
        final_spins = np.squeeze(np.array(samples[-1]))
        portfolio = (final_spins + 1) / 2  # 0 or 1

        # Compute portfolio metrics
        selected_assets = np.where(portfolio == 1)[0]
        total_return = h[selected_assets].sum() if len(selected_assets) > 0 else 0
        risk = 0
        if len(selected_assets) > 1:
            cov = corr_matrix[np.ix_(selected_assets, selected_assets)]
            risk = np.sum(cov) / 2  # Simplified variance

        st.success("MCMC Completed!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Acceptance Rate", f"{acc_rate:.3f}")
            st.metric("Final Energy", f"{energies[-1]:.3f}")
        with col2:
            st.metric("Selected Assets", len(selected_assets))
            st.metric("Total Return", f"{total_return:.3f}")
            st.metric("Portfolio Risk", f"{risk:.3f}")

        st.subheader("Energy Trace")
        fig, ax = plt.subplots()
        ax.plot(energies)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.set_title("MCMC Energy Trace")
        st.pyplot(fig)

        st.subheader("Final Portfolio")
        st.write(f"Selected Assets: {selected_assets.tolist()}")
        st.write(f"Portfolio Vector: {portfolio.astype(int)}")

        st.subheader("Sample Statistics")
        st.write(f"Mean Spin: {sample_mean:.3f}")
        st.write(f"Std Dev Spin: {sample_stddev:.3f}")
