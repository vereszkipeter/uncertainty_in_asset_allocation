# app.py

import streamlit as st
import pandas as pd
import numpy as np
import h5py
import plotly.graph_objects as go
from scipy.optimize import minimize
import os

# --- Oldal Konfiguráció ---
st.set_page_config(
    page_title="Eszközallokáció Bizonytalansága",
    layout="wide"
)

# --- KONFIGURÁCIÓ ÉS ÁLLANDÓK ---
RISK_FREE_RATE = 0.02
DATA_DIR = "data"
HDF5_FILE = os.path.join(DATA_DIR, "asset_allocation_uncertainty.h5")

COLORS = {
    'frontier': '#4169E1',   # royalblue
    'assets': '#000000',     # black
    'user': '#DC143C',       # crimson
    'min_risk': '#32CD32',   # limegreen
    'tangency': '#9400D3',   # darkviolet
    'risk_parity': '#FF8C00' # darkorange
}

# --- SZÁMÍTÁSI FÜGGVÉNYEK ---
# Ezek a függvények szükségesek a pontbecslés (base case) és
# a felhasználói portfólió metrikáinak valós idejű számításához.

def calculate_portfolio_cvar(weights, monthly_returns, confidence_level=0.95):
    """Kiszámítja egy portfólió évesített CVaR (Expected Shortfall) értékét."""
    weights = np.array(weights)
    portfolio_returns = monthly_returns.dot(weights)
    var = np.quantile(portfolio_returns, 1 - confidence_level)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar * 12

def calculate_portfolio_return(weights, expected_returns):
    """Kiszámítja egy portfólió évesített várható hozamát."""
    return np.sum(np.array(weights) * expected_returns)

def get_portfolio_metrics(weights, expected_returns, monthly_returns):
    """Egyben adja vissza a portfólió kockázati (CVaR) és hozam metrikáit."""
    return calculate_portfolio_cvar(weights, monthly_returns), calculate_portfolio_return(weights, expected_returns)

@st.cache_data
def calculate_efficient_frontier(_expected_returns, _monthly_returns, n_points=20):
    """Kiszámítja a hatékony front pontjait. Cache-elve a gyorsaságért."""
    results = []
    target_returns = np.linspace(_expected_returns.min(), _expected_returns.max(), n_points)
    bounds = tuple((0.0, 1.0) for _ in range(len(_expected_returns)))
    initial_weights = np.full(len(_expected_returns), 1.0 / len(_expected_returns))
    for target_ret in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'eq', 'fun': lambda w: calculate_portfolio_return(w, _expected_returns) - target_ret})
        res = minimize(calculate_portfolio_cvar, initial_weights, args=(_monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success: results.append({'cvar': res.fun, 'ret': target_ret})
    if not results: return pd.DataFrame(columns=['cvar', 'ret'])
    return pd.DataFrame(results).sort_values(by='cvar').drop_duplicates()

@st.cache_data
def calculate_special_portfolios(_expected_returns, _monthly_returns, risk_free_rate):
    """Kiszámítja a kiemelt portfóliókat. Cache-elve a gyorsaságért."""
    # A minimize-hoz szükséges segédfüggvények
    def neg_sharpe_ratio_cvar(weights, expected_returns, monthly_returns, risk_free_rate):
        p_cvar, p_ret = get_portfolio_metrics(weights, expected_returns, monthly_returns)
        if p_cvar <= 1e-6: return np.inf
        return -(p_ret - risk_free_rate) / p_cvar

    def risk_parity_cvar_objective(weights, monthly_returns):
        weights = np.array(weights)
        portfolio_returns = monthly_returns.dot(weights)
        var_threshold = np.quantile(portfolio_returns, 0.05)
        tail_scenarios = monthly_returns[portfolio_returns <= var_threshold]
        if tail_scenarios.empty: return 1e6
        component_expected_shortfall = -tail_scenarios.mean(axis=0).values * 12
        risk_contributions = weights * component_expected_shortfall
        target_contribution = np.sum(risk_contributions) / len(weights)
        return np.sum((risk_contributions - target_contribution)**2)

    num_assets = len(_expected_returns)
    initial_weights = np.full(num_assets, 1.0 / num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    min_risk_res = minimize(calculate_portfolio_cvar, initial_weights, args=(_monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    tangency_res = minimize(neg_sharpe_ratio_cvar, initial_weights, args=(_expected_returns, _monthly_returns, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    risk_parity_res = minimize(risk_parity_cvar_objective, initial_weights, args=(_monthly_returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    return {
        'min_risk': min_risk_res.x if min_risk_res.success else initial_weights,
        'tangency': tangency_res.x if tangency_res.success else initial_weights,
        'risk_parity': risk_parity_res.x if risk_parity_res.success else initial_weights
    }

# --- ADATBETÖLTÉS ÉS FELDOLGOZÁS ---

@st.cache_resource
def load_data_from_hdf5(file_path):
    """Betölti és feldolgozza az összes adatot a HDF5 fájlból."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        data['title'] = f.attrs.get('title', "Eszközallokáció Bizonytalansága")
        
        # Alap adatok
        asset_names = [name.decode('utf-8') for name in f['base/asset_names'][:]]
        dates = pd.to_datetime(f['base/dates'][:])
        data['base_returns'] = pd.DataFrame(f['base/returns'][:], index=dates, columns=asset_names)
        data['asset_classes'] = asset_names
        
        # Szimulációs kimenetek (pontfelhők, frontok)
        outputs = f['simulation_outputs']
        data['sim_min_risk_points'] = pd.DataFrame(outputs['min_risk_points'][:], columns=['cvar', 'ret'])
        data['sim_tangency_points'] = pd.DataFrame(outputs['tangency_points'][:], columns=['cvar', 'ret'])
        data['sim_risk_parity_points'] = pd.DataFrame(outputs['risk_parity_points'][:], columns=['cvar', 'ret'])
        
        data['sim_frontiers'] = []
        for name in outputs['efficient_frontiers']:
            df = pd.DataFrame(outputs['efficient_frontiers'][name][:], columns=['cvar', 'ret'])
            data['sim_frontiers'].append(df)

        # Szimulációs bemenetek (a felhasználói portfólió felhőjéhez)
        inputs = f['simulation_inputs']
        data['sim_expected_returns'] = inputs['expected_returns'][:]
        
        bs_cols = [col.decode('utf-8') for col in inputs['bootstrap_samples_columns'][:]]
        bs_data = inputs['bootstrap_samples'][:]
        all_samples_df = pd.DataFrame(bs_data, columns=bs_cols)
        
        # A dátumot visszaalakítjuk, és szétbontjuk a nagy táblát egy listává
        all_samples_df['date'] = pd.to_datetime(all_samples_df['date'])
        data['bootstrap_samples_list'] = [
            df.drop(columns=['sim_id', 'date']).set_index(df['date'].values)
            for _, df in all_samples_df.groupby('sim_id')
        ]
    return data

@st.cache_data
def calculate_user_portfolio_cloud(_weights_tuple, sim_ers, bootstrap_samples_list):
    """Kiszámítja a felhasználói portfólió pontfelhőjét a szimulált bemenetek alapján."""
    weights_array = np.array([w for _, w in _weights_tuple])
    cloud_points = [
        get_portfolio_metrics(weights_array, sim_er, bootstrap_sample)
        for sim_er, bootstrap_sample in zip(sim_ers, bootstrap_samples_list)
    ]
    return pd.DataFrame(cloud_points, columns=['cvar', 'ret'])

# --- ALKALMAZÁS FELÉPÍTÉSE ---

try:
    all_data = load_data_from_hdf5(HDF5_FILE)
except FileNotFoundError:
    st.error(f"Az adatfájl ({HDF5_FILE}) nem található! Kérlek, futtasd először a `prepare_data.py` szkriptet.")
    st.stop()

st.title(all_data['title'])

# Adatok előkészítése
returns_data = all_data['base_returns']
ASSET_CLASSES = all_data['asset_classes']
base_er = returns_data.mean()

# Pontbecslés számítása (csak egyszer fut le a cache miatt)
with st.spinner("Alapértelmezett portfóliók és hatékony front számítása..."):
    base_frontier = calculate_efficient_frontier(base_er, returns_data)
    base_special_weights = calculate_special_portfolios(base_er, returns_data, RISK_FREE_RATE)
    
    base_assets_metrics = [get_portfolio_metrics(np.identity(len(ASSET_CLASSES))[i], base_er, returns_data) for i in range(len(ASSET_CLASSES))]
    base_assets_df = pd.DataFrame(base_assets_metrics, columns=['cvar', 'ret'])
    
    base_min_risk_metrics = get_portfolio_metrics(base_special_weights['min_risk'], base_er, returns_data)
    base_tangency_metrics = get_portfolio_metrics(base_special_weights['tangency'], base_er, returns_data)
    base_risk_parity_metrics = get_portfolio_metrics(base_special_weights['risk_parity'], base_er, returns_data)

# --- Oldalsáv (Sidebar) ---
st.sidebar.header("A portfólió súlyai")
weights = {asset: st.sidebar.slider(label=asset, min_value=0.0, max_value=1.0, value=1./len(ASSET_CLASSES), step=0.01) for asset in ASSET_CLASSES}
total_w = sum(weights.values())
norm_weights = {asset: w / total_w if total_w > 0 else 1./len(ASSET_CLASSES) for asset, w in weights.items()}
weights_vector = np.array([norm_weights[asset] for asset in ASSET_CLASSES])

st.sidebar.subheader("Aktuális súlyok")
for asset, weight in norm_weights.items():
    st.sidebar.write(f"{asset}: {weight:.2%}")

# --- Felhasználói portfólió számításai ---
user_portfolio_cvar, user_portfolio_return = get_portfolio_metrics(weights_vector, base_er, returns_data)
weights_tuple_for_cache = tuple(sorted(norm_weights.items()))
user_cloud_df = calculate_user_portfolio_cloud(weights_tuple_for_cache, all_data['sim_expected_returns'], all_data['bootstrap_samples_list'])

# --- Főoldali megjelenítés ---
st.subheader("A portfólió jellemzői (pontbecslés alapján)")
cols = st.columns(2)
cols[0].metric("Évesített várható hozam", f"{user_portfolio_return * 100:.2f}%")
cols[1].metric("Évesített CVaR (95%)", f"{user_portfolio_cvar * 100:.2f}%")

st.subheader("Portfóliók a kockázat (CVaR) - hozam térben")
fig = go.Figure()

def to_rgba(hex_color, alpha): 
    return f"rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)}, {alpha})"

# 1. Szimulált felhők és frontok (halvány háttér)
for i, fr in enumerate(all_data['sim_frontiers']):
    fig.add_trace(go.Scatter(x=fr['cvar'], y=fr['ret'], mode='lines', line_color=to_rgba(COLORS['frontier'], 0.1), name='Szimulált front', legendgroup='sim', showlegend=(i == 0)))
for key in ['min_risk', 'tangency', 'risk_parity']:
    df = all_data[f'sim_{key}_points']
    fig.add_trace(go.Scatter(x=df['cvar'], y=df['ret'], mode='markers', marker=dict(color=to_rgba(COLORS[key], 0.2), size=5), name=f'Szimulált {key.replace("_", " ")}', legendgroup='sim'))
fig.add_trace(go.Scatter(x=user_cloud_df['cvar'], y=user_cloud_df['ret'], mode='markers', marker=dict(color=to_rgba(COLORS['user'], 0.3), size=5), name='Az Ön portfóliójának szimulációi', legendgroup='sim'))

# 2. Pontbecslés (vastag vonalak és nagy pontok)
fig.add_trace(go.Scatter(x=base_frontier['cvar'], y=base_frontier['ret'], mode='lines', line=dict(color=COLORS['frontier'], width=3), name='Hatékony front (pontbecslés)'))
fig.add_trace(go.Scatter(x=base_assets_df['cvar'], y=base_assets_df['ret'], mode='markers+text', marker=dict(size=12, color=COLORS['assets'], symbol='diamond'), text=[name.split('(')[0].strip() for name in ASSET_CLASSES], textposition="middle right", name="Egyedi eszközök"))
fig.add_trace(go.Scatter(x=[base_min_risk_metrics[0]], y=[base_min_risk_metrics[1]], mode='markers', marker=dict(size=15, color=COLORS['min_risk'], symbol='circle', line=dict(width=1, color='black')), name="Min Risk portfólió"))
fig.add_trace(go.Scatter(x=[base_tangency_metrics[0]], y=[base_tangency_metrics[1]], mode='markers', marker=dict(size=15, color=COLORS['tangency'], symbol='square', line=dict(width=1, color='black')), name="Tangency portfólió"))
fig.add_trace(go.Scatter(x=[base_risk_parity_metrics[0]], y=[base_risk_parity_metrics[1]], mode='markers', marker=dict(size=15, color=COLORS['risk_parity'], symbol='triangle-up', line=dict(width=1, color='black')), name="Risk Parity portfólió"))
fig.add_trace(go.Scatter(x=[user_portfolio_cvar], y=[user_portfolio_return], mode='markers', marker=dict(size=18, color=COLORS['user'], symbol='star', line=dict(width=1, color='black')), name="Az Ön portfóliója"))

# Grafikon beállításai
fig.update_layout(
    xaxis_title='Évesített CVaR (95%) - Kockázat', 
    yaxis_title='Évesített várható hozam',
    yaxis_tickformat=".2%", 
    xaxis_tickformat=".2%", 
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01), 
    height=700,
    title='A portfólióallokáció bizonytalansága a kockázat-hozam térben'
)
st.plotly_chart(fig, use_container_width=True)
