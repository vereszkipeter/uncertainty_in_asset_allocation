"""Streamlit dashboard: uncertainty in asset allocation.

The app visualizes point estimates and bootstrap uncertainty in the
CVaR-return plane. All portfolio math lives in :mod:`portfolio` so that the
data pipeline and the app use exactly the same definitions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sibling modules importable when Streamlit runs this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CONFIDENCE_LEVEL,
    HDF5_FILE,
    N_FRONTIER_POINTS,
    RISK_FREE_RATE,
    TITLE,
)
from data_io import load_results  # noqa: E402
from portfolio import (  # noqa: E402
    annualized_mean,
    efficient_frontier,
    optimize_special_portfolios,
    portfolio_metrics,
    portfolio_return,
)

# --- Page configuration -----------------------------------------------------
st.set_page_config(page_title="Eszközallokáció Bizonytalansága", layout="wide")

COLORS = {
    "frontier": "#4169E1",
    "assets": "#D3D3D3",
    "user": "#DC143C",
    "min_risk": "#32CD32",
    "tangency": "#9400D3",
    "risk_parity": "#FF8C00",
}


# --- Helpers ----------------------------------------------------------------
def format_weights_for_tooltip(weights, asset_names) -> str:
    """Format portfolio weights as an HTML block for Plotly tooltips."""
    header = "<br><b>Portfólió súlyok:</b><br>"
    lines = [f"{name}: {weight:.2%}" for name, weight in zip(asset_names, weights)]
    return header + "<br>".join(lines)


def to_rgba(hex_color: str, alpha: float) -> str:
    return (
        f"rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, "
        f"{int(hex_color[5:7], 16)}, {alpha})"
    )


def calculate_user_portfolio_cloud(
    weights_vector, sim_ers, bootstrap_samples_list, confidence: float
) -> pd.DataFrame:
    """Simulated CVaR-return cloud for the user's portfolio."""
    cloud_points = [
        portfolio_metrics(weights_vector, sim_er, sample, confidence)
        for sim_er, sample in zip(sim_ers, bootstrap_samples_list)
    ]
    return pd.DataFrame(cloud_points, columns=["cvar", "ret"])


@st.cache_data(show_spinner=False)
def cached_frontier(
    _expected_returns, _monthly_returns, n_points, confidence, min_return
):
    return efficient_frontier(
        _expected_returns,
        _monthly_returns,
        n_points=n_points,
        confidence=confidence,
        min_return=min_return,
    )


@st.cache_data(show_spinner=False)
def cached_special_portfolios(
    _expected_returns, _monthly_returns, risk_free_rate, confidence
):
    return optimize_special_portfolios(
        _expected_returns,
        _monthly_returns,
        risk_free_rate,
        confidence=confidence,
    )


@st.cache_resource
def load_data(file_path):
    return load_results(file_path)


# --- Data -------------------------------------------------------------------
try:
    all_data = load_data(HDF5_FILE)
except Exception as exc:
    st.error(
        f"Hiba a '{HDF5_FILE}' beolvasása közben: '{exc}'. "
        "Kérlek, futtasd a `python src/prepare_data.py` parancsot az adatok "
        "generálásához."
    )
    st.stop()

returns_data = all_data["base_returns"]
asset_classes = all_data["asset_classes"]
confidence = float(all_data.get("confidence_level", CONFIDENCE_LEVEL))
risk_free_rate = float(all_data.get("risk_free_rate", RISK_FREE_RATE))
base_er = annualized_mean(returns_data)

st.title(all_data.get("title") or TITLE)
st.caption(
    f"{returns_data.index.min():%Y-%m} – {returns_data.index.max():%Y-%m} | "
    f"{len(asset_classes)} eszköz | "
    f"{all_data.get('n_simulations', 0)} bootstrap szimuláció | "
    f"{confidence:.0%} CVaR | seed={all_data.get('seed', 'n/a')}"
)

# --- Point-estimate portfolios ---------------------------------------------
with st.spinner("Alap portfóliók és hatékony front számítása..."):
    base_special = cached_special_portfolios(
        base_er, returns_data, risk_free_rate, confidence
    )
    base_special_weights = {
        key: result.weights for key, result in base_special.items()
    }
    base_frontier = cached_frontier(
        base_er,
        returns_data,
        N_FRONTIER_POINTS,
        confidence,
        portfolio_return(base_special_weights["min_risk"], base_er),
    )
    base_metrics = {
        key: portfolio_metrics(weights, base_er, returns_data, confidence)
        for key, weights in base_special_weights.items()
    }
    identity = np.eye(len(asset_classes))
    base_assets_df = pd.DataFrame(
        [
            portfolio_metrics(identity[index], base_er, returns_data, confidence)
            for index in range(len(asset_classes))
        ],
        columns=["cvar", "ret"],
    )

# --- Sidebar ----------------------------------------------------------------
st.sidebar.header("A portfólió súlyai")
if st.sidebar.button("Egyenlő súlyok visszaállítása", use_container_width=True):
    for asset in asset_classes:
        st.session_state[f"weight_{asset}"] = 1.0 / len(asset_classes)

weights = {
    asset: st.sidebar.slider(
        f"{index + 1}. {asset}",
        0.0,
        1.0,
        1.0 / len(asset_classes),
        0.01,
        key=f"weight_{asset}",
    )
    for index, asset in enumerate(asset_classes)
}

total_w = sum(weights.values())
if total_w > 0:
    norm_weights = {asset: weight / total_w for asset, weight in weights.items()}
else:
    norm_weights = {
        asset: 1.0 / len(asset_classes) for asset in asset_classes
    }
weights_vector = np.array(
    [norm_weights[asset] for asset in asset_classes], dtype=float
)

st.sidebar.subheader("Aktuális súlyok (normalizált)")
for asset, weight in norm_weights.items():
    st.sidebar.write(f"{asset}: {weight:.2%}")

st.sidebar.header("Modell bizonytalanság")
show_simulations = st.sidebar.checkbox(
    "Bizonytalanság megjelenítése", value=False
)
if show_simulations:
    st.sidebar.info(
        "A pontbecslések mellett a bootstrap szimulált eredmények "
        "(halványabb pontfelhők és vonalak) is láthatók a grafikonon."
    )

# --- Main metrics -----------------------------------------------------------
user_cvar, user_ret = portfolio_metrics(
    weights_vector, base_er, returns_data, confidence
)
st.subheader("A portfólió jellemzői (pontbecslés alapján)")
metric_columns = st.columns(2)
metric_columns[0].metric("Évesített várható hozam", f"{user_ret * 100:.2f}%")
metric_columns[1].metric(
    f"Havi CVaR ({confidence:.0%})", f"{user_cvar * 100:.2f}%"
)

# --- Chart ------------------------------------------------------------------
st.subheader("Portfóliók a kockázat (CVaR) - hozam térben")
figure = go.Figure()

if show_simulations:
    for index, frontier in enumerate(all_data["sim_frontiers"]):
        figure.add_trace(
            go.Scatter(
                x=frontier["cvar"],
                y=frontier["ret"],
                mode="lines",
                line_color=to_rgba(COLORS["frontier"], 0.1),
                name="Szimulált frontok",
                legendgroup="sim_frontiers",
                showlegend=(index == 0),
            )
        )

    for key in ("min_risk", "tangency", "risk_parity"):
        points = all_data[f"sim_{key}_points"]
        sim_weights_matrix = all_data[f"sim_{key}_weights"]
        hover_texts = [
            format_weights_for_tooltip(weights, asset_classes)
            for weights in sim_weights_matrix
        ]
        figure.add_trace(
            go.Scatter(
                x=points["cvar"],
                y=points["ret"],
                text=hover_texts,
                mode="markers",
                marker=dict(color=to_rgba(COLORS[key], 0.2), size=7),
                name=f'Szimulált {key.replace("_", " ")}',
                legendgroup=f"sim_{key}",
                hovertemplate=(
                    f"<b>Szimulált {key.replace('_', ' ')}</b><br><br>"
                    "Kockázat (CVaR): %{x:.2%}<br>"
                    "Várható Hozam: %{y:.2%}%{text}<extra></extra>"
                ),
            )
        )

    user_cloud_df = calculate_user_portfolio_cloud(
        weights_vector,
        all_data["sim_expected_returns"],
        all_data["bootstrap_samples_list"],
        confidence,
    )
    if not user_cloud_df.empty:
        user_weights_text_cloud = format_weights_for_tooltip(
            weights_vector, asset_classes
        )
        figure.add_trace(
            go.Scatter(
                x=user_cloud_df["cvar"],
                y=user_cloud_df["ret"],
                customdata=[user_weights_text_cloud] * len(user_cloud_df),
                mode="markers",
                marker=dict(color=to_rgba(COLORS["user"], 0.3), size=7),
                name="Az Ön portfóliójának szimulációi",
                legendgroup="sim_user",
                hovertemplate=(
                    "<b>Szimulált pont (Ön)</b><br><br>"
                    "Kockázat (CVaR): %{x:.2%}<br>"
                    "Várható Hozam: %{y:.2%}%{customdata}<extra></extra>"
                ),
            )
        )

figure.add_trace(
    go.Scatter(
        x=base_frontier["cvar"],
        y=base_frontier["ret"],
        mode="lines",
        line=dict(color=COLORS["frontier"], width=3),
        name="Hatékony front (pontbecslés)",
    )
)

for index, asset_name in enumerate(asset_classes):
    asset_label = re.sub(r"\s*\([^)]*\)", "", asset_name)
    asset_weights = identity[index]
    weights_text_asset = format_weights_for_tooltip(asset_weights, asset_classes)
    figure.add_trace(
        go.Scatter(
            x=[base_assets_df["cvar"].iloc[index]],
            y=[base_assets_df["ret"].iloc[index]],
            mode="markers+text",
            marker=dict(
                size=12,
                color=COLORS["assets"],
                symbol="diamond",
                line=dict(color="black", width=1),
            ),
            text=asset_label,
            textposition="middle right",
            name="Egyedi eszközök",
            legendgroup="assets",
            showlegend=(index == 0),
            customdata=[weights_text_asset],
            hovertemplate=(
                f"<b>{asset_name}</b><br><br>"
                "Kockázat (CVaR): %{x:.2%}<br>"
                "Várható Hozam: %{y:.2%}%{customdata}<extra></extra>"
            ),
        )
    )

portfolio_labels = {
    "min_risk": "Min Risk",
    "tangency": "Tangency",
    "risk_parity": "Risk Parity",
}
symbols = {"min_risk": "circle", "tangency": "square", "risk_parity": "triangle-up"}
for key, label in portfolio_labels.items():
    cvar, ret = base_metrics[key]
    weights_text = format_weights_for_tooltip(
        base_special_weights[key], asset_classes
    )
    figure.add_trace(
        go.Scatter(
            x=[cvar],
            y=[ret],
            text=[weights_text],
            mode="markers",
            marker=dict(
                size=15,
                color=COLORS[key],
                symbol=symbols[key],
                line=dict(width=1, color="black"),
            ),
            name=f"{label} portfólió",
            hovertemplate=(
                f"<b>{label} portfólió</b><br><br>"
                "Kockázat (CVaR): %{x:.2%}<br>"
                "Várható Hozam: %{y:.2%}%{text}<extra></extra>"
            ),
        )
    )

weights_text_user = format_weights_for_tooltip(weights_vector, asset_classes)
figure.add_trace(
    go.Scatter(
        x=[user_cvar],
        y=[user_ret],
        text=[weights_text_user],
        mode="markers",
        marker=dict(
            size=18,
            color=COLORS["user"],
            symbol="star",
            line=dict(width=1, color="black"),
        ),
        name="Az Ön portfóliója",
        hovertemplate=(
            "<b>Az Ön portfóliója</b><br><br>"
            "Kockázat (CVaR): %{x:.2%}<br>"
            "Várható Hozam: %{y:.2%}%{text}<extra></extra>"
        ),
    )
)

figure.update_layout(
    xaxis_title=f"Havi CVaR ({confidence:.0%}) - Kockázat",
    yaxis_title="Évesített várható hozam",
    yaxis_tickformat=".2%",
    xaxis_tickformat=".2%",
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    height=700,
    title="A portfólióallokáció bizonytalansága a kockázat-hozam térben",
)
st.plotly_chart(figure, use_container_width=True)

# --- Weights table and methodology -----------------------------------------
with st.expander("Portfóliósúlyok táblázata"):
    weights_table = pd.DataFrame(
        {
            "Eszköz": asset_classes,
            "Az Ön portfóliója": weights_vector,
            "Min Risk": base_special_weights["min_risk"],
            "Tangency": base_special_weights["tangency"],
            "Risk Parity": base_special_weights["risk_parity"],
        }
    )
    st.dataframe(
        weights_table.style.format(
            {
                column: "{:.2%}"
                for column in weights_table.columns
                if column != "Eszköz"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Módszertan és korlátok"):
    st.markdown(
        f"""
        - **Adatok:** havi hozamok {returns_data.index.min():%Y-%m} és
          {returns_data.index.max():%Y-%m} között, {len(asset_classes)} ETF-ből.
        - **Bizonytalanság:** *joint bootstrap*. Minden szimulációban
          újramintavételezzük a havi hozamokat, majd ugyanabból a mintából
          számítjuk a várható hozamot, a CVaR-t és a hatékony frontot.
        - **Várható hozam:** a bootstrap minta átlaga × 12 (évesítve).
        - **Kockázat:** havi empirikus CVaR (Expected Shortfall),
          {confidence:.0%}-os konfidenciaszinten.
        - **Megkötések:** hosszú pozíciók, teljes befektetés, tranzakciós
          költség nélkül.
        - **Korlátok:** i.i.d. bootstrap (nincs idősor-függőség), in-sample
          becslés, konstans kockázatmentes hozam, kis eszközuniverzum.
        """
    )
