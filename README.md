# Pricing an Exotic Basket Option with Monte Carlo & Recurrent Neural Networks

**University challenge · Mondragon Unibertsitatea × BBVA (Global Markets) · Sep–Nov 2026**
Team project (6 members) – BSc Business Data Analytics, 3rd year

> Public copy of the team's original (private) repository, shared for portfolio purposes. All credit for the code goes to the team listed below.

---

## The problem

Many of the products traded by BBVA's Global Markets desk have no closed-form price and are valued with Monte Carlo simulation, which is slow, especially when a product has to be revalued many times during its life.

We studied an **Asian (geometric-average) call option on a basket of three Dow Jones stocks** and asked:
*can a neural network trained on Monte Carlo simulations price the option as accurately as the simulation itself, and do it instantly?*

## What we did

| Step | Description |
|---|---|
| **1. Data** | Built a clean dataset of 30 Dow Jones stocks (Yahoo Finance): cleaning, stock-split adjustment, time alignment. |
| **2. Basket selection** | Scored 1,656 possible 3-stock combinations on return, risk and diversification (Sharpe, Sortino, drawdown, correlation). Selected basket: **Walmart (WMT), Merck (MRK) and NVIDIA (NVDA)**. |
| **3. Time series & volatility** | Stationarity tests and volatility modelling of each stock, used to calibrate the simulation. |
| **4. Monte Carlo** | Correlated geometric Brownian motion, 20 quarters, **100,000 simulated paths**. Premium: **0.3014**. |
| **5. Recurrent neural networks** | Trained simple RNN, LSTM and GRU models to predict the option payoff directly from the price path (20 quarters × 3 stocks), with hyperparameter search. |
| **6. Business case** | How the model could speed up mid-life revaluations and the quoting of new baskets for the Global Markets desk. |

## Key results

- The final **2-layer GRU (64 units)** reached **R² = 0.99999** on the test set.
- It reproduced the Monte Carlo premium within **0.05 %** — about 18 times smaller than the Monte Carlo's own 95 % margin of error.
- **99 %** of the prediction errors fall between −0.003 and +0.005.
- When part of the price path is missing (valuing the option mid-life), the current network underestimates the premium (−2.9 % with one quarter missing, −10.5 % with four) → next step: train on truncated paths.

## Tech stack

Python · pandas · NumPy · statsmodels · arch · SciPy · PyTorch · yfinance · Matplotlib · Plotly · Jupyter

## How to run

Notebooks run in this order (each uses the output of the previous one):

1. `Preprocesamiento_limpieza.ipynb` – data cleaning
2. `EDA.py` – exploratory analysis
3. `seleccioncartera.ipynb` – basket selection
4. `Analisis_series_volatilidad_final.ipynb` – time series & volatility
5. `montecarlo.ipynb` – Monte Carlo simulation
6. `rnn.ipynb` – neural network training (~3.5 h)
7. `resultados.ipynb` – final comparison

Install dependencies with `python instalar_librerias.py`. Raw data is not included in the repository (as required by the challenge rules).

## Team

Gabriel Elorriaga · Ander Sainz de la Maza · Ander Duñabeitia · Iñigo Barrenechea · Mario Abajas · Álvaro Moreno

*Code comments and notebooks are in Spanish.*
