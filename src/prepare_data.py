# prepare_data.py (Javított, KeyError nélkül)

import pandas as pd
import numpy as np
import yfinance as yf
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

TICKERS = {
    "USD Készpénz (BIL)": "BIL", 
    "USA Állampapír (IEF)": "IEF",
    "USA Váll. Kötvény (LQD)": "LQD", 
    "Arany (GLD)": "GLD"
}
ASSET_CLASSES = list(TICKERS.keys())
TICKER_LIST = list(TICKERS.values())

# --- ADATLETÖLTŐ FÜGGVÉNY ---

def load_data(tickers, start_date, end_date):
    print("Adatok letöltése a yfinance segítségével...")
    try:
        full_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        price_data = full_data['Close']
        if price_data.empty:
            raise ValueError("Az adatok letöltése üres DataFrame-et eredményezett.")
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])
        
        returns = price_data.resample('M').ffill().pct_change().dropna()
        returns.columns = ASSET_CLASSES
        print("Adatletöltés sikeres.")
        return returns
    except Exception as e:
        print(f"Hiba az adatletöltés során: {e}")
        return pd.DataFrame()

# --- PORTFÓLIÓ-OPTIMALIZÁCIÓS FÜGGVÉNYEK ---
# (Ezek a függvények változatlanok)
def calculate_portfolio_cvar(weights, monthly_returns, confidence_level=0.95):
    weights = np.array(weights)
    portfolio_returns = monthly_returns.dot(weights)
    var = np.quantile(portfolio_returns, 1 - confidence_level)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar * 12

def calculate_portfolio_return(weights, expected_returns):
    return np.sum(np.array(weights) * expected_returns)

def get_portfolio_metrics(weights, expected_returns, monthly_returns):
    return calculate_portfolio_cvar(weights, monthly_returns), calculate_portfolio_return(weights, expected_returns)

def neg_sharpe_ratio_cvar(weights, expected_returns, monthly_returns, risk_free_rate):
    p_cvar, p_ret = get_portfolio_metrics(weights, expected_returns, monthly_returns)
    if p_cvar <= 1e-6: return np.inf
    return -(p_ret - risk_free_rate) / p_cvar

def risk_parity_cvar_objective(weights, monthly_returns, confidence_level=0.95):
    weights = np.array(weights)
    portfolio_returns = monthly_returns.dot(weights)
    var_threshold = np.quantile(portfolio_returns, 1 - confidence_level)
    tail_scenarios = monthly_returns[portfolio_returns <= var_threshold]
    if tail_scenarios.empty: return 1e6
    component_expected_shortfall = -tail_scenarios.mean(axis=0).values * 12
    risk_contributions = weights * component_expected_shortfall
    target_contribution = np.sum(risk_contributions) / len(weights)
    return np.sum((risk_contributions - target_contribution)**2)

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
    
    returns_data = load_data(TICKER_LIST, START_DATE, END_DATE)
    if returns_data.empty:
        print("Az adatletöltés sikertelen, a program leáll.")
        exit()

    print(f"\nIndul a {N_SIMULATIONS} db Monte Carlo szimuláció...")
    num_obs, num_assets = returns_data.shape
    base_er = returns_data.mean().values * 12
    er_cov = np.diag(((returns_data.std() / np.sqrt(num_obs)) * np.sqrt(12))**2)

    sim_inputs = {'expected_returns': [], 'covariance_matrices': [], 'bootstrap_samples': []}
    sim_outputs = {'frontiers': [], 'min_risk_points': [], 'tangency_points': [], 'risk_parity_points': []}

    for i in range(N_SIMULATIONS):
        print(f"  - Feldolgozás: {i+1}/{N_SIMULATIONS}")
        sim_er = np.random.multivariate_normal(mean=base_er, cov=er_cov)
        bootstrap_sample = returns_data.sample(n=num_obs, replace=True)
        sim_inputs['expected_returns'].append(sim_er)
        sim_inputs['covariance_matrices'].append(bootstrap_sample.cov().values * 12)
        sim_inputs['bootstrap_samples'].append(bootstrap_sample)
        
        sim_outputs['frontiers'].append(calculate_efficient_frontier(sim_er, bootstrap_sample))
        special_weights = calculate_special_portfolios(sim_er, bootstrap_sample, RISK_FREE_RATE)
        sim_outputs['min_risk_points'].append(get_portfolio_metrics(special_weights['min_risk'], sim_er, bootstrap_sample))
        sim_outputs['tangency_points'].append(get_portfolio_metrics(special_weights['tangency'], sim_er, bootstrap_sample))
        sim_outputs['risk_parity_points'].append(get_portfolio_metrics(special_weights['risk_parity'], sim_er, bootstrap_sample))

    print("\nSzimulációk sikeresen lefutottak.")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Eredmények mentése a(z) '{OUTPUT_FILE}' fájlba...")
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.attrs['title'] = "Az eszközallokáció input adataiban rejlő bizonytalanság"
        
        base_group = f.create_group('base')
        base_group.create_dataset('returns', data=returns_data.to_numpy())
        base_group.create_dataset('dates', data=returns_data.index.astype(np.int64))
        base_group.create_dataset('asset_names', data=ASSET_CLASSES)

        inputs_group = f.create_group('simulation_inputs')
        inputs_group.create_dataset('expected_returns', data=np.array(sim_inputs['expected_returns']))
        inputs_group.create_dataset('covariance_matrices', data=np.array(sim_inputs['covariance_matrices']))
        
        all_samples_df = pd.concat(
            sim_inputs['bootstrap_samples'], 
            keys=range(N_SIMULATIONS),
            names=['sim_id']
        ).reset_index()

        # JAVÍTÁS: A reset_index() az eredeti (név nélküli) dátum indexből 'index' nevű oszlopot csinál.
        # Ezt nevezzük át 'date'-re.
        all_samples_df = all_samples_df.rename(columns={'index': 'date'})
        
        # Most már létezik a 'date' oszlop, így az átalakítás működni fog.
        all_samples_df['date'] = all_samples_df['date'].astype(np.int64)
        
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