# prepare_data.py (Végleges, javított Risk Parity logikával)

import pandas as pd
import numpy as np
import yfinance
from scipy.optimize import minimize
import h5py
import warnings
import time
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- KONFIGURÁCIÓ ---
RISK_FREE_RATE = 0.02
N_SIMULATIONS = 100
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "asset_allocation_uncertainty.h5")
RETURNS_CSV_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")

TICKERS = {
    "USD Készpénz (BIL)": "BIL", 
    "USA Állampapír (IEF)": "IEF",
    "USA Váll. Kötvény (LQD)": "LQD", 
    "Arany (GLD)": "GLD"
}
ASSET_CLASSES = list(TICKERS.keys())
TICKER_LIST = list(TICKERS.values())

# --- ADATLETÖLTŐ ÉS CACHELŐ FÜGGVÉNY ---
def get_returns_data(csv_path, tickers, start_date, end_date):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"Hozamadatok betöltése a helyi cache-ből: '{csv_path}'")
        try:
            return pd.read_csv(csv_path, index_col='date', parse_dates=True)
        except Exception as e:
            print(f"Hiba a CSV fájl olvasása közben: {e}. Újra letöltjük az adatokat.")
    
    print("Helyi cache nem található. Adatok letöltése a yfinance segítségével...")
    try:
        full_data = yfinance.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        returns_df = full_data['Close'].resample('M').ffill().pct_change().dropna()
        if isinstance(returns_df, pd.Series):
             returns_df = returns_df.to_frame(name=ASSET_CLASSES[0])
        else:
            returns_df.columns = ASSET_CLASSES
        returns_df.index.name = 'date'
        returns_df.to_csv(csv_path)
        print(f"Adatok sikeresen lementve a cache-be: '{csv_path}'")
        return returns_df
    except Exception as e:
        print(f"Hiba az adatletöltés során: {e}")
        return pd.DataFrame()

# --- PORTFÓLIÓ-OPTIMALIZÁCIÓS FÜGGVÉNYEK ---
def calculate_portfolio_cvar(weights, monthly_returns):
    portfolio_returns = monthly_returns.dot(np.array(weights))
    var = np.quantile(portfolio_returns, 0.05)
    return -portfolio_returns[portfolio_returns <= var].mean() * 12

def calculate_portfolio_return(weights, expected_returns):
    return np.sum(np.array(weights) * expected_returns)

def get_portfolio_metrics(weights, expected_returns, monthly_returns):
    return calculate_portfolio_cvar(weights, monthly_returns), calculate_portfolio_return(weights, expected_returns)

def neg_sharpe_ratio_cvar(weights, expected_returns, monthly_returns, risk_free_rate):
    p_cvar, p_ret = get_portfolio_metrics(weights, expected_returns, monthly_returns)
    if p_cvar <= 1e-6: return np.inf
    return -(p_ret - risk_free_rate) / p_cvar

# JAVÍTÁS 2: Robusztus Risk Parity célfüggvény ide is bekerül
def risk_parity_cvar_objective(weights, monthly_returns):
    weights = np.array(weights)
    num_assets = len(weights)
    portfolio_returns = monthly_returns.dot(weights)
    var_threshold = np.quantile(portfolio_returns, 0.05)
    tail_scenarios = monthly_returns[portfolio_returns <= var_threshold]
    if tail_scenarios.empty: return 1e6
    component_expected_shortfall = -tail_scenarios.mean(axis=0).values * 12
    risk_contributions = weights * component_expected_shortfall
    total_risk = np.sum(risk_contributions)
    if total_risk <= 1e-6: return 1e6
    relative_contributions = risk_contributions / total_risk
    target_contributions = np.full(num_assets, 1.0 / num_assets)
    return np.sum((relative_contributions - target_contributions)**2)

def calculate_special_portfolios(expected_returns, monthly_returns, risk_free_rate):
    num_assets = len(expected_returns)
    initial_weights = np.full(num_assets, 1.0 / num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    min_risk_res = minimize(calculate_portfolio_cvar, initial_weights, args=(monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    tangency_res = minimize(neg_sharpe_ratio_cvar, initial_weights, args=(expected_returns, monthly_returns, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    risk_parity_res = minimize(risk_parity_cvar_objective, initial_weights, args=(monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    return {
        'min_risk': min_risk_res.x if min_risk_res.success else initial_weights,
        'tangency': tangency_res.x if tangency_res.success else initial_weights,
        'risk_parity': risk_parity_res.x if risk_parity_res.success else initial_weights
    }

def calculate_efficient_frontier(expected_returns, monthly_returns, n_points=20):
    results = []
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    bounds = tuple((0.0, 1.0) for _ in range(len(expected_returns)))
    initial_weights = np.full(len(expected_returns), 1.0 / len(expected_returns))
    for target_ret in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'eq', 'fun': lambda w: calculate_portfolio_return(w, expected_returns) - target_ret})
        res = minimize(calculate_portfolio_cvar, initial_weights, args=(monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success: results.append({'cvar': res.fun, 'ret': target_ret})
    if not results: return pd.DataFrame(columns=['cvar', 'ret'])
    return pd.DataFrame(results).sort_values(by='cvar').drop_duplicates()

# --- FŐ VÉGREHAJTÓ BLOKK ---
if __name__ == "__main__":
    start_time = time.time()
    returns_data = get_returns_data(RETURNS_CSV_FILE, TICKER_LIST, START_DATE, END_DATE)
    if returns_data.empty: exit()
    print(f"\nIndul a {N_SIMULATIONS} db Monte Carlo szimuláció...")
    num_obs, num_assets = returns_data.shape
    base_er = returns_data.mean().values * 12
    er_cov = np.diag(((returns_data.std() / np.sqrt(num_obs)) * np.sqrt(12))**2)
    sim_inputs = {'expected_returns': [], 'bootstrap_samples': []}
    sim_outputs = {'frontiers': [], 'min_risk_points': [], 'tangency_points': [], 'risk_parity_points': []}
    for i in range(N_SIMULATIONS):
        print(f"  - Feldolgozás: {i+1}/{N_SIMULATIONS}")
        sim_er = np.random.multivariate_normal(mean=base_er, cov=er_cov)
        bootstrap_sample = returns_data.sample(n=num_obs, replace=True)
        sim_inputs['expected_returns'].append(sim_er)
        sim_inputs['bootstrap_samples'].append(bootstrap_sample)
        sim_outputs['frontiers'].append(calculate_efficient_frontier(sim_er, bootstrap_sample))
        special_weights = calculate_special_portfolios(sim_er, bootstrap_sample, RISK_FREE_RATE)
        sim_outputs['min_risk_points'].append(get_portfolio_metrics(special_weights['min_risk'], sim_er, bootstrap_sample))
        sim_outputs['tangency_points'].append(get_portfolio_metrics(special_weights['tangency'], sim_er, bootstrap_sample))
        sim_outputs['risk_parity_points'].append(get_portfolio_metrics(special_weights['risk_parity'], sim_er, bootstrap_sample))
    print("\nSzimulációk sikeresen lefutottak.")
    print(f"Eredmények mentése a(z) '{OUTPUT_FILE}' fájlba...")
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.attrs['title'] = "Az eszközallokáció input adataiban rejlő bizonytalanság"
        returns_data.index.name = 'date'
        base_group = f.create_group('base')
        base_group.create_dataset('returns', data=returns_data.to_numpy())
        base_group.create_dataset('dates', data=returns_data.index.astype(np.int64))
        base_group.create_dataset('asset_names', data=ASSET_CLASSES)
        inputs_group = f.create_group('simulation_inputs')
        inputs_group.create_dataset('expected_returns', data=np.array(sim_inputs['expected_returns']))
        all_samples_df = pd.concat(sim_inputs['bootstrap_samples'], keys=range(N_SIMULATIONS), names=['sim_id']).reset_index()
        date_column_name = all_samples_df.columns[1]
        all_samples_df[date_column_name] = all_samples_df[date_column_name].astype(np.int64)
        all_samples_df.rename(columns={date_column_name: 'date'}, inplace=True)
        inputs_group.create_dataset('bootstrap_samples', data=all_samples_df.to_numpy())
        inputs_group.create_dataset('bootstrap_samples_columns', data=all_samples_df.columns.tolist())
        outputs_group = f.create_group('simulation_outputs')
        outputs_group.create_dataset('min_risk_points', data=np.array(sim_outputs['min_risk_points']))
        outputs_group.create_dataset('tangency_points', data=np.array(sim_outputs['tangency_points']))
        outputs_group.create_dataset('risk_parity_points', data=np.array(sim_outputs['risk_parity_points']))
        frontiers_group = outputs_group.create_group('efficient_frontiers')
        for i, df in enumerate(sim_outputs['frontiers']):
            if not df.empty:
                frontiers_group.create_dataset(f'frontier_{i}', data=df.to_numpy())
    end_time = time.time()
    print(f"\nMinden adat sikeresen elmentve. Futási idő: {end_time - start_time:.2f} másodperc.")