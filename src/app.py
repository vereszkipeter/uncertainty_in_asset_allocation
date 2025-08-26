# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- Oldal Konfiguráció ---
st.set_page_config(page_title="Robusztus eszközallokáció", layout="wide")

# --- KONFIGURÁCIÓ ÉS SZÍNBEÁLLÍTÁSOK ---
RISK_FREE_RATE = 0.02
N_SIMULATIONS = 100
COLORS = {
    'frontier': '#4169E1',   # royalblue
    'assets': '#000000',     # black
    'user': '#DC143C',       # crimson
    'min_risk': '#32CD32',   # limegreen
    'tangency': '#9400D3',   # darkviolet
    'risk_parity': '#FF8C00' # darkorange
}

# --- SZÁMÍTÁSI FÜGGVÉNYEK ---

@st.cache_data
def load_data(tickers, start_date, end_date):
    try:
        full_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        price_data = full_data.get('Adj Close', full_data.get('Close'))
        if price_data is None or price_data.empty: return pd.DataFrame()
        if isinstance(price_data, pd.Series): price_data = price_data.to_frame(name=tickers[0])
        return price_data.resample('M').ffill().pct_change().dropna()
    except Exception:
        return pd.DataFrame()

def calculate_portfolio_cvar(weights, monthly_returns, confidence_level=0.95):
    portfolio_returns = monthly_returns.dot(np.array(weights))
    var = portfolio_returns.quantile(1 - confidence_level)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar * 12

def calculate_portfolio_return(weights, expected_returns):
    return np.sum(np.array(weights) * expected_returns)

def get_portfolio_metrics(weights, expected_returns, monthly_returns):
    cvar = calculate_portfolio_cvar(weights, monthly_returns)
    ret = calculate_portfolio_return(weights, expected_returns)
    return cvar, ret

def risk_parity_cvar_objective(weights, monthly_returns, confidence_level=0.95):
    """JAVÍTOTT, robusztus célfüggvény a CVaR Parity portfólióhoz."""
    weights = np.array(weights)
    num_assets = len(weights)
    portfolio_returns = monthly_returns.dot(weights)
    var_threshold = portfolio_returns.quantile(1 - confidence_level)
    
    tail_scenarios = monthly_returns[portfolio_returns <= var_threshold]
    if tail_scenarios.empty: return 1e6 # Nagy hiba, ha nincs farok

    component_expected_returns_in_tail = -tail_scenarios.mean(axis=0).values * 12
    risk_contributions = weights * component_expected_returns_in_tail
    
    total_risk = np.sum(risk_contributions)
    if total_risk <= 0: return 1e6
    
    relative_contributions = risk_contributions / total_risk
    target_contributions = np.full(num_assets, 1.0 / num_assets)
    
    # A cél a relatív hozzájárulások és az ideális 1/N közötti négyzetes eltérés minimalizálása
    return np.sum((relative_contributions - target_contributions)**2)

def neg_sharpe_ratio_cvar(weights, expected_returns, monthly_returns, risk_free_rate):
    cvar, ret = get_portfolio_metrics(weights, expected_returns, monthly_returns)
    if cvar <= 0: return np.inf
    return -(ret - risk_free_rate) / cvar

@st.cache_data
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

@st.cache_data
def calculate_efficient_frontier(expected_returns, monthly_returns, n_points=20):
    results = []
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    bounds = tuple((0.0, 1.0) for _ in range(len(expected_returns)))
    for target_ret in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'eq', 'fun': lambda w: calculate_portfolio_return(w, expected_returns) - target_ret})
        res = minimize(calculate_portfolio_cvar, [1./len(expected_returns)]*len(expected_returns), args=(monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success: results.append({'cvar': res.fun, 'ret': target_ret})
    return pd.DataFrame(results).sort_values(by='cvar')

@st.cache_data
def run_monte_carlo_simulation(_returns_data, risk_free_rate, n_simulations=N_SIMULATIONS):
    results = {'frontiers': [], 'asset_points': [], 'min_risk_points': [], 'tangency_points': [], 'risk_parity_points': [], 'scenarios': []}
    num_obs, num_assets = _returns_data.shape
    base_er = _returns_data.mean() * 12
    er_cov = np.diag(((_returns_data.std() / np.sqrt(num_obs)) * np.sqrt(12))**2)
    for _ in range(n_simulations):
        bootstrap_sample = _returns_data.sample(n=num_obs, replace=True)
        sim_er = np.random.multivariate_normal(mean=base_er, cov=er_cov)
        results['scenarios'].append({'er': sim_er, 'returns': bootstrap_sample})
        results['frontiers'].append(calculate_efficient_frontier(sim_er, bootstrap_sample, n_points=15))
        sim_assets_cvar = [calculate_portfolio_cvar([1 if i==j else 0 for j in range(num_assets)], bootstrap_sample) for i in range(num_assets)]
        results['asset_points'].append({'cvar': sim_assets_cvar, 'ret': sim_er})
        special_weights = calculate_special_portfolios(sim_er, bootstrap_sample, risk_free_rate)
        results['min_risk_points'].append(get_portfolio_metrics(special_weights['min_risk'], sim_er, bootstrap_sample))
        results['tangency_points'].append(get_portfolio_metrics(special_weights['tangency'], sim_er, bootstrap_sample))
        results['risk_parity_points'].append(get_portfolio_metrics(special_weights['risk_parity'], sim_er, bootstrap_sample))
    return results

# --- ALKALMAZÁS FELÉPÍTÉSE ---
st.title("Eszközallokáció: paraméterbecslési hiba hatás")

TICKERS = {"USD Készpénz (BIL)": "BIL", "USA Állampapír (IEF)": "IEF", "USA Váll. Kötvény (LQD)": "LQD", "Arany (GLD)": "GLD"}
ASSET_CLASSES = list(TICKERS.keys())

returns_data = load_data(list(TICKERS.values()), "2010-01-01", "2024-12-31")
if returns_data.empty: st.error("Adatok letöltése sikertelen."); st.stop()
returns_data.columns = ASSET_CLASSES

base_er = returns_data.mean() * 12
with st.spinner("Alapértelmezett hatékony front és portfóliók számítása..."):
    base_frontier = calculate_efficient_frontier(base_er, returns_data)
    base_special_weights = calculate_special_portfolios(base_er, returns_data, RISK_FREE_RATE)
    base_assets_cvar = [get_portfolio_metrics([1 if i==j else 0 for j in range(len(ASSET_CLASSES))], base_er, returns_data)[0] for i in range(len(ASSET_CLASSES))]
    base_min_risk_metrics = get_portfolio_metrics(base_special_weights['min_risk'], base_er, returns_data)
    base_tangency_metrics = get_portfolio_metrics(base_special_weights['tangency'], base_er, returns_data)
    base_risk_parity_metrics = get_portfolio_metrics(base_special_weights['risk_parity'], base_er, returns_data)

st.sidebar.header("A portfólió súlyai")
weights = {asset: st.sidebar.slider(label=asset, min_value=0.0, max_value=1.0, value=1./len(ASSET_CLASSES), step=0.01) for asset in ASSET_CLASSES}
total_w = sum(weights.values())
norm_weights = {asset: w / total_w if total_w > 0 else 1./len(ASSET_CLASSES) for asset, w in weights.items()}
weights_vector = np.array([norm_weights[asset] for asset in ASSET_CLASSES])

st.sidebar.subheader("Aktuális súlyok")
# JAVÍTÁS: Egyszerű for ciklus a list comprehension helyett
for asset, weight in norm_weights.items():
    st.sidebar.write(f"{asset}: {weight:.2%}")

st.sidebar.header("Modell bizonytalanság"); show_uncertainty = st.sidebar.checkbox("Bizonytalanság megjelenítése")
if show_uncertainty: st.sidebar.info(f"A pontbecslések mellett **{N_SIMULATIONS} szimulációt** is megjelenítünk.")

sim_data = None
if show_uncertainty:
    with st.spinner("Monte Carlo szimuláció futtatása..."):
        sim_data = run_monte_carlo_simulation(returns_data, RISK_FREE_RATE)
    user_portfolio_cloud = [get_portfolio_metrics(weights_vector, s['er'], s['returns']) for s in sim_data['scenarios']]

user_portfolio_cvar, user_portfolio_return = get_portfolio_metrics(weights_vector, base_er, returns_data)

st.subheader("A portfólió jellemzői (pontbecslés alapján)"); cols = st.columns(3)
cols[0].metric("Évesített várható hozam", f"{user_portfolio_return:.2%}")
cols[1].metric("Évesített CVaR (95%)", f"{user_portfolio_cvar:.2%}")

st.subheader("Portfóliók a kockázat (CVaR) - hozam térben"); fig = go.Figure()

if show_uncertainty and sim_data:
    def to_rgba(hex_color, alpha): return f"rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)}, {alpha})"
    for i, fr in enumerate(sim_data['frontiers']): fig.add_trace(go.Scatter(x=fr['cvar'], y=fr['ret'], mode='lines', line_color=to_rgba(COLORS['frontier'], 0.1), legendgroup='sim_frontiers', name='Szimulált frontok' if i == 0 else '', showlegend=(i == 0)))
    sim_asset_pts = pd.DataFrame([item for sub in [list(zip(d['cvar'], d['ret'])) for d in sim_data['asset_points']] for item in sub], columns=['cvar', 'ret'])
    fig.add_trace(go.Scatter(x=sim_asset_pts['cvar'], y=sim_asset_pts['ret'], mode='markers', marker=dict(color=to_rgba(COLORS['assets'], 0.2), size=5), name='Szimulált eszközpontok'))
    for key in ['min_risk', 'tangency', 'risk_parity']:
        cloud = pd.DataFrame(sim_data[f'{key}_points'], columns=['cvar','ret'])
        fig.add_trace(go.Scatter(x=cloud['cvar'], y=cloud['ret'], mode='markers', marker=dict(color=to_rgba(COLORS[key], 0.2), size=5), name=f'Szimulált {key.replace("_", " ")} portfóliók'))
    user_cloud_df = pd.DataFrame(user_portfolio_cloud, columns=['cvar', 'ret'])
    fig.add_trace(go.Scatter(x=user_cloud_df['cvar'], y=user_cloud_df['ret'], mode='markers', marker=dict(color=to_rgba(COLORS['user'], 0.3), size=5), name='Az Ön portfóliójának szimulációi'))

fig.add_trace(go.Scatter(x=base_frontier['cvar'], y=base_frontier['ret'], mode='lines', line=dict(color=COLORS['frontier'], width=3), name='Hatékony front (pontbecslés)'))
fig.add_trace(go.Scatter(x=base_assets_cvar, y=base_er, mode='markers+text', marker=dict(size=12, color=COLORS['assets'], symbol='diamond'), text=[name.split('(')[0] for name in ASSET_CLASSES], textposition="middle right", name="Egyedi eszközök"))
fig.add_trace(go.Scatter(x=[base_min_risk_metrics[0]], y=[base_min_risk_metrics[1]], mode='markers', marker=dict(size=15, color=COLORS['min_risk'], symbol='circle', line=dict(width=1, color='black')), name="Min Risk portfólió"))
fig.add_trace(go.Scatter(x=[base_tangency_metrics[0]], y=[base_tangency_metrics[1]], mode='markers', marker=dict(size=15, color=COLORS['tangency'], symbol='square', line=dict(width=1, color='black')), name="Tangency portfólió"))
fig.add_trace(go.Scatter(x=[base_risk_parity_metrics[0]], y=[base_risk_parity_metrics[1]], mode='markers', marker=dict(size=15, color=COLORS['risk_parity'], symbol='triangle-up', line=dict(width=1, color='black')), name="Risk Parity portfólió"))
fig.add_trace(go.Scatter(x=[user_portfolio_cvar], y=[user_portfolio_return], mode='markers', marker=dict(size=18, color=COLORS['user'], symbol='star', line=dict(width=1, color='black')), name="Az Ön portfóliója"))

fig.update_layout(xaxis_title='Évesített CVaR (95%) - Kockázat', yaxis_title='Évesített várható hozam', yaxis_tickformat=".2%", xaxis_tickformat=".2%", legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01), height=700, title='A portfólióallokáció bizonytalansága a kockázat-hozam térben')
st.plotly_chart(fig, use_container_width=True)
