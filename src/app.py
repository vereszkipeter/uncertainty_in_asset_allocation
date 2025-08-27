# app.py (Végleges, garantáltan működő verzió)

import streamlit as st
import pandas as pd
import numpy as np
import h5py
import plotly.graph_objects as go
from scipy.optimize import minimize
import os

# --- Oldal Konfiguráció ---
st.set_page_config(page_title="Eszközallokáció Bizonytalansága", layout="wide")

# --- KONFIGURÁCIÓ ÉS ÁLLANDÓK ---
RISK_FREE_RATE = 0.02
DATA_DIR = "data"
HDF5_FILE = os.path.join(DATA_DIR, "asset_allocation_uncertainty.h5")

COLORS = {
    'frontier': '#4169E1', 'assets': '#D3D3D3', 'user': '#DC143C',
    'min_risk': '#32CD32', 'tangency': '#9400D3', 'risk_parity': '#FF8C00'
}

# --- SZÁMÍTÁSI ÉS SEGÉDFÜGGVÉNYEK ---
def calculate_portfolio_cvar(weights, monthly_returns):
    portfolio_returns = monthly_returns.dot(np.array(weights))
    var = np.quantile(portfolio_returns, 0.05)
    return -portfolio_returns[portfolio_returns <= var].mean() * 12

def calculate_portfolio_return(weights, expected_returns):
    return np.sum(np.array(weights) * expected_returns)

def get_portfolio_metrics(weights, expected_returns, monthly_returns):
    return calculate_portfolio_cvar(weights, monthly_returns), calculate_portfolio_return(weights, expected_returns)

def format_weights_for_tooltip(weights, asset_names):
    text = "<br><b>Portfólió súlyok:</b><br>"
    text += "<br>".join([f"{name}: {weight:.2%}" for name, weight in zip(asset_names, weights)])
    return text

@st.cache_data
def calculate_efficient_frontier(_expected_returns, _monthly_returns):
    results = []
    target_returns = np.linspace(_expected_returns.min(), _expected_returns.max(), 20)
    bounds = tuple((0.0, 1.0) for _ in range(len(_expected_returns)))
    initial_weights = np.full(len(_expected_returns), 1.0 / len(_expected_returns))
    for target_ret in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'eq', 'fun': lambda w: calculate_portfolio_return(w, _expected_returns) - target_ret})
        res = minimize(calculate_portfolio_cvar, initial_weights, args=(_monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success: results.append({'cvar': res.fun, 'ret': target_ret})
    return pd.DataFrame(results).sort_values(by='cvar').drop_duplicates() if results else pd.DataFrame(columns=['cvar', 'ret'])

@st.cache_data
def calculate_special_portfolios(_expected_returns, _monthly_returns, risk_free_rate):
    def neg_sharpe_ratio_cvar(weights, er, mr, rfr):
        p_cvar, p_ret = get_portfolio_metrics(weights, er, mr)
        return -(p_ret - rfr) / p_cvar if p_cvar > 1e-6 else np.inf
    def risk_parity_cvar_objective(weights, mr):
        weights, num_assets = np.array(weights), len(weights)
        portfolio_returns = mr.dot(weights)
        var_threshold = np.quantile(portfolio_returns, 0.05)
        tail_scenarios = mr[portfolio_returns <= var_threshold]
        if tail_scenarios.empty: return 1e6
        risk_contributions = weights * (-tail_scenarios.mean(axis=0).values * 12)
        total_risk = np.sum(risk_contributions)
        if total_risk <= 1e-6: return 1e6
        return np.sum(((risk_contributions / total_risk) - (1/num_assets))**2)
    num_assets = len(_expected_returns)
    initial_weights = np.full(num_assets, 1.0 / num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    min_risk_res = minimize(calculate_portfolio_cvar, initial_weights, args=(_monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    tangency_res = minimize(neg_sharpe_ratio_cvar, initial_weights, args=(_expected_returns, _monthly_returns, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    risk_parity_res = minimize(risk_parity_cvar_objective, initial_weights, args=(_monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    return { 'min_risk': min_risk_res.x, 'tangency': tangency_res.x, 'risk_parity': risk_parity_res.x }

@st.cache_resource
def load_data_from_hdf5(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        data['title'] = f.attrs.get('title', "Eszközallokáció Bizonytalansága")
        asset_names = [name.decode('utf-8') for name in f['base/asset_names'][:]]
        dates = pd.to_datetime(f['base/dates'][:])
        data['base_returns'] = pd.DataFrame(f['base/returns'][:], index=dates, columns=asset_names)
        data['asset_classes'] = asset_names
        outputs = f['simulation_outputs']
        for key in ['min_risk', 'tangency', 'risk_parity']:
            data[f'sim_{key}_points'] = pd.DataFrame(outputs[f'{key}_points'][:], columns=['cvar', 'ret'])
        data['sim_frontiers'] = [pd.DataFrame(outputs['efficient_frontiers'][name][:], columns=['cvar', 'ret']) for name in outputs['efficient_frontiers']]
        inputs = f['simulation_inputs']
        data['sim_expected_returns'] = inputs['expected_returns'][:]
        bs_cols = [col.decode('utf-8') for col in inputs['bootstrap_samples_columns'][:]]
        all_samples_df = pd.DataFrame(inputs['bootstrap_samples'][:], columns=bs_cols)
        all_samples_df['date'] = pd.to_datetime(all_samples_df['date'])
        data['bootstrap_samples_list'] = [df.set_index('date').drop(columns='sim_id') for _, df in all_samples_df.groupby('sim_id')]
    return data

# JAVÍTÁS: A @st.cache_data dekorátor eltávolítása innen
def calculate_user_portfolio_cloud(weights_vector, sim_ers, bootstrap_samples_list):
    """Ez a függvény már nincs cache-elve, hogy mindig friss eredményt adjon."""
    cloud_points = [get_portfolio_metrics(weights_vector, sim_er, bootstrap_sample) for sim_er, bootstrap_sample in zip(sim_ers, bootstrap_samples_list)]
    return pd.DataFrame(cloud_points, columns=['cvar', 'ret'])

# --- ALKALMAZÁS FELÉPÍTÉSE ---
try:
    all_data = load_data_from_hdf5(HDF5_FILE)
except FileNotFoundError:
    st.error(f"Az adatfájl ({HDF5_FILE}) nem található! Kérlek, futtasd először a `prepare_data.py` szkriptet.")
    st.stop()
st.title(all_data['title'])
returns_data = all_data['base_returns']
ASSET_CLASSES = all_data['asset_classes']
base_er = returns_data.mean().values * 12

with st.spinner("Alapértelmezett portfóliók és hatékony front számítása..."):
    base_frontier = calculate_efficient_frontier(base_er, returns_data)
    base_special_weights = calculate_special_portfolios(base_er, returns_data, RISK_FREE_RATE)
    base_metrics = {key: get_portfolio_metrics(w, base_er, returns_data) for key, w in base_special_weights.items()}
    base_assets_metrics = [get_portfolio_metrics(np.identity(len(ASSET_CLASSES))[i], base_er, returns_data) for i in range(len(ASSET_CLASSES))]
    base_assets_df = pd.DataFrame(base_assets_metrics, columns=['cvar', 'ret'])

# --- Oldalsáv (Sidebar) ---
st.sidebar.header("A portfólió súlyai")
weights = {asset: st.sidebar.slider(f, 0.0, 1.0, 1./len(ASSET_CLASSES), 0.01) for f, asset in zip([f'{i+1}. {asset}' for i, asset in enumerate(ASSET_CLASSES)], ASSET_CLASSES)}
total_w = sum(weights.values())
norm_weights = {asset: w / total_w if total_w > 0 else 1./len(ASSET_CLASSES) for asset, w in weights.items()}
weights_vector = np.array(list(norm_weights.values()))

st.sidebar.subheader("Aktuális súlyok")
for asset, weight in norm_weights.items(): st.sidebar.write(f"{asset}: {weight:.2%}")

st.sidebar.header("Modell bizonytalanság")
show_simulations = st.sidebar.checkbox("Bizonytalanság megjelenítése", value=False)
if show_simulations: st.sidebar.info("A pontbecslések mellett a szimulált eredmények is láthatók.")

# --- Főoldali megjelenítés ---
user_cvar, user_ret = get_portfolio_metrics(weights_vector, base_er, returns_data)
st.subheader("A portfólió jellemzői (pontbecslés alapján)")
cols = st.columns(2)
cols[0].metric("Évesített várható hozam", f"{user_ret * 100:.2f}%")
cols[1].metric("Évesített CVaR (95%)", f"{user_cvar * 100:.2f}%")

st.subheader("Portfóliók a kockázat (CVaR) - hozam térben")
fig = go.Figure()
def to_rgba(hex, alpha): return f"rgba({int(hex[1:3], 16)}, {int(hex[3:5], 16)}, {int(hex[5:7], 16)}, {alpha})"

if show_simulations:
    user_cloud_df = calculate_user_portfolio_cloud(weights_vector, all_data['sim_expected_returns'], all_data['bootstrap_samples_list'])
    for i, fr in enumerate(all_data['sim_frontiers']):
        fig.add_trace(go.Scatter(x=fr['cvar'], y=fr['ret'], mode='lines', line_color=to_rgba(COLORS['frontier'], 0.1), name='Szimulált frontok', legendgroup='sim_frontiers', showlegend=(i == 0)))
    for key in ['min_risk', 'tangency', 'risk_parity']:
        df = all_data[f'sim_{key}_points']
        fig.add_trace(go.Scatter(x=df['cvar'], y=df['ret'], mode='markers', marker=dict(color=to_rgba(COLORS[key], 0.2), size=7), name=f'Szimulált {key.replace("_", " ")}', legendgroup=f'sim_{key}'))
    if not user_cloud_df.empty:
        fig.add_trace(go.Scatter(x=user_cloud_df['cvar'], y=user_cloud_df['ret'], mode='markers', marker=dict(color=to_rgba(COLORS['user'], 0.3), size=7), name='Az Ön portfóliójának szimulációi', legendgroup='sim_user'))

# --- Pontbecslések és tooltipek (mindig látszanak) ---
fig.add_trace(go.Scatter(x=base_frontier['cvar'], y=base_frontier['ret'], mode='lines', line=dict(color=COLORS['frontier'], width=3), name='Hatékony front (pontbecslés)'))
for i, asset_name in enumerate(ASSET_CLASSES):
    fig.add_trace(go.Scatter(
        x=[base_assets_df['cvar'][i]], y=[base_assets_df['ret'][i]], mode='markers+text',
        marker=dict(size=12, color=COLORS['assets'], symbol='diamond', line=dict(color='black', width=1)),
        text=asset_name.split('(')[0].strip(), textposition="middle right", name="Egyedi eszközök", legendgroup="assets", showlegend=(i==0),
        hovertemplate=f"<b>{asset_name}</b><br><br>Kockázat (CVaR): %{{x:.2%}}<br>Várható Hozam: %{{y:.2%}}<extra></extra>"
    ))
portfolio_map = {'min_risk': 'Min Risk', 'tangency': 'Tangency', 'risk_parity': 'Risk Parity'}
symbols = {'min_risk': 'circle', 'tangency': 'square', 'risk_parity': 'triangle-up'}
for key, name in portfolio_map.items():
    cvar, ret = base_metrics[key]
    weights_text = format_weights_for_tooltip(base_special_weights[key], ASSET_CLASSES)
    fig.add_trace(go.Scatter(
        x=[cvar], y=[ret], mode='markers', marker=dict(size=15, color=COLORS[key], symbol=symbols[key], line=dict(width=1, color='black')), name=f"{name} portfólió",
        text=weights_text,
        hovertemplate=f"<b>{name} portfólió</b><br><br>Kockázat (CVaR): %{{x:.2%}}<br>Várható Hozam: %{{y:.2%}}%{{text}}<extra></extra>"
    ))
weights_text_user = format_weights_for_tooltip(weights_vector, ASSET_CLASSES)
fig.add_trace(go.Scatter(
    x=[user_cvar], y=[user_ret], mode='markers', marker=dict(size=18, color=COLORS['user'], symbol='star', line=dict(width=1, color='black')), name="Az Ön portfóliója",
    text=weights_text_user,
    hovertemplate=f"<b>Az Ön portfóliója</b><br><br>Kockázat (CVaR): %{{x:.2%}}<br>Várható Hozam: %{{y:.2%}}%{{text}}<extra></extra>"
))

fig.update_layout(
    xaxis_title='Évesített CVaR (95%) - Kockázat', yaxis_title='Évesített várható hozam',
    yaxis_tickformat=".2%", xaxis_tickformat=".2%", legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    height=700, title='A portfólióallokáció bizonytalansága a kockázat-hozam térben'
)
st.plotly_chart(fig, use_container_width=True)