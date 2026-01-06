# Quantum-enhanced MCMC for Portfolio Optimization

This repository contains code for applying quantum-enhanced Markov Chain Monte Carlo (MCMC) methods to portfolio optimization problems, modeled as Ising models.
l streamlit
## Files

- `qmcmc.py`: Core implementation of quantum and classical MCMC kernels for Ising models.
- `Clique_qmcmc.ipynb`: Jupyter notebook demonstrating the method on maximum clique problems.
- `qmcmc_requirements.txt`: Python dependencies.
- `app.py`: Streamlit web app for interactive portfolio optimization demo.

## Portfolio Optimization Mapping

In the Ising model:
- Spins (+1/-1) represent asset inclusion/exclusion.
- J matrix: Negative correlations to penalize correlated assets (risk).
- h vector: Expected returns to favor high-return assets.
- Energy minimization finds optimal portfolios.

## Running the Demo Locally

1. Install dependencies:
   ```bash
   pip install -r qmcmc_requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Create a GitHub repository and push this code.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select the repository.
4. Set the main file path to `app.py`.
5. Set the requirements file to `qmcmc_requirements.txt`.
6. Deploy!

## Deploying to Heroku

1. Install Heroku CLI and login.
2. Create a Heroku app:
   ```bash
   heroku create your-app-name
   ```
3. Add buildpacks:
   ```bash
   heroku buildpacks:add --index 1 heroku/python
   heroku buildpacks:add --index 2 https://github.com/heroku/heroku-buildpack-apt
   ```
4. Push to Heroku:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```
5. Open the app:
   ```bash
   heroku open
   ```

Note: Heroku may have issues with TensorFlow Quantum; Streamlit Cloud is recommended.

## Usage

In the web app:
- Set number of assets.
- Input expected returns (h).
- Input correlation matrix (J, will be negated for Ising).
- Choose kernel: quantum (uses TFQ), classical (uniform), local (single flip).
- Run MCMC and view results: energy trace, final portfolio, metrics.

## Notes

- Quantum kernel uses sparse edges for efficiency.
- Requires TensorFlow Quantum and Cirq for quantum simulation.
- For real quantum hardware, modify the sampling layer.
