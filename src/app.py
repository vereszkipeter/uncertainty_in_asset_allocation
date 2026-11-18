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
    APP_TITLE,
    CACHE_VERSION,
    CONFIDENCE_LEVEL,
    HDF5_FILE,
    N_FRONTIER_POINTS,
    RISK_FREE_RATE,
)
from data_io import load_results  # noqa: E402
from portfolio import (  # noqa: E402
    annualized_mean,
    efficient_frontier,
    optimize_special_portfolios,
    portfolio_metrics,
    portfolio_return,
)
from ui_helpers import (  # noqa: E402
    build_metrics_table,
    build_weights_table,
    format_weights_for_tooltip,
    frontier_insight,
    ticker_label,
    to_rgba,
)

# --- Page configuration -----------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
)

# The default Streamlit h1 is large enough to wrap the Hungarian title on a
# laptop viewport, pushing the chart below the fold. A slightly smaller title
# keeps it on one line without changing the document structure.
st.markdown(
    """
    <style>
    h1 {
        font-size: 1.75rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Okabe-Ito colorblind-safe palette.
COLORS = {
    "frontier": "#0072B2",
    "assets": "#7F7F7F",
    "user": "#D55E00",
    "min_risk": "#009E73",
    "tangency": "#CC79A7",
    "risk_parity": "#56B4E9",
}
PORTFOLIO_NAMES = {
    "min_risk": "Min Risk",
    "tangency": "Tangency",
    "risk_parity": "Risk Parity",
}
SIMULATION_SYMBOLS = {
    "min_risk": "circle",
    "tangency": "square",
    "risk_parity": "triangle-up",
    "user": "star",
}
SIMULATION_LABELS = {
    "min_risk": "Min Risk",
    "tangency": "Tangency",
    "risk_parity": "Risk Parity",
    "user": "Ön portfóliója",
}


# --- Helpers ----------------------------------------------------------------
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
    _expected_returns,
    _monthly_returns,
    n_points,
    confidence,
    min_return,
    cache_version,
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
    _expected_returns,
    _monthly_returns,
    risk_free_rate,
    confidence,
    cache_version,
):
    return optimize_special_portfolios(
        _expected_returns,
        _monthly_returns,
        risk_free_rate,
        confidence=confidence,
    )


@st.cache_resource
def load_data(file_path, file_mtime):
    return load_results(file_path)


# --- Data -------------------------------------------------------------------
try:
    all_data = load_data(HDF5_FILE, HDF5_FILE.stat().st_mtime_ns)
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

st.title(APP_TITLE)
st.caption(
    f"{returns_data.index.min():%Y-%m} – {returns_data.index.max():%Y-%m} | "
    f"{len(asset_classes)} eszköz | "
    f"{all_data.get('n_simulations', 0)} bootstrap szimuláció | "
    f"{confidence:.0%} CVaR | seed={all_data.get('seed', 'n/a')}"
)

# --- Point-estimate portfolios ---------------------------------------------
with st.spinner("Alap portfóliók és hatékony front számítása..."):
    base_special = cached_special_portfolios(
        base_er, returns_data, risk_free_rate, confidence, CACHE_VERSION
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
        CACHE_VERSION,
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
# The simulation switch is intentionally the first control: it used to sit
# below four sliders and fell below the fold on a laptop viewport.
st.sidebar.header("Minta bizonytalanság")
show_simulations = st.sidebar.toggle(
    "Bootstrap felhők megjelenítése",
    value=False,
    help=(
        "A pontbecslések mellett a 100 bootstrap minta szimulált "
        "portfólióit is megjeleníti a grafikonon."
    ),
)
if show_simulations:
    st.sidebar.caption(
        "A halvány pontok és frontok bootstrap mintákból származnak; "
        "a szimbólumok kulcsa a chart bal felső sarkában van."
    )

st.sidebar.divider()
st.sidebar.header("A portfólió súlyai")


def reset_weights() -> None:
    """Reset slider state from the button callback.

    Streamlit forbids assigning a widget-backed session-state key after the
    widget has been instantiated. An ``on_click`` callback runs before the
    script and therefore before the sliders are created.
    """
    equal_weight = 1.0 / len(asset_classes)
    for asset in asset_classes:
        st.session_state[f"weight_{asset}"] = equal_weight


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
    norm_weights = {asset: 1.0 / len(asset_classes) for asset in asset_classes}
weights_vector = np.array(
    [norm_weights[asset] for asset in asset_classes], dtype=float
)

st.sidebar.caption("Normalizált súlyok")
st.sidebar.markdown(
    " · ".join(
        f"**{ticker_label(asset)}** {norm_weights[asset]:.1%}"
        for asset in asset_classes
    )
)
st.sidebar.button(
    "Egyenlő súlyok visszaállítása",
    width="stretch",
    on_click=reset_weights,
)

# --- Current portfolio summary and chart -----------------------------------
user_cvar, user_ret = portfolio_metrics(
    weights_vector, base_er, returns_data, confidence
)
st.caption(
    f"Az Ön portfóliója: **{user_ret:.2%}** évesített hozam · "
    f"**{user_cvar:.2%}** havi CVaR ({confidence:.0%})"
)
st.subheader("Kockázat–hozam tér")
figure = go.Figure()

if show_simulations:
    for index, frontier in enumerate(all_data["sim_frontiers"]):
        figure.add_trace(
            go.Scatter(
                x=frontier["cvar"],
                y=frontier["ret"],
                mode="lines",
                line=dict(color=to_rgba(COLORS["frontier"], 0.12), width=1),
                hoverinfo="skip",
                name="Bootstrap frontok",
                legendgroup="bootstrap",
                legendrank=10,
                showlegend=False,
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
                marker=dict(
                    color=to_rgba(COLORS[key], 0.42),
                    size=8.5,
                    symbol=SIMULATION_SYMBOLS[key],
                    line=dict(color=to_rgba(COLORS[key], 0.95), width=0.8),
                ),
                name=f"Bootstrap: {PORTFOLIO_NAMES[key]}",
                legendgroup=f"bootstrap_{key}",
                showlegend=False,
                hovertemplate=(
                    f"<b>Bootstrap: {PORTFOLIO_NAMES[key]}</b><br><br>"
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
                marker=dict(
                    color=to_rgba(COLORS["user"], 0.45),
                    size=9,
                    symbol="star",
                    line=dict(color=to_rgba(COLORS["user"], 0.95), width=0.8),
                ),
                name="Bootstrap: Az Ön portfóliója",
                legendgroup="bootstrap_user",
                showlegend=False,
                hovertemplate=(
                    "<b>Bootstrap: Az Ön portfóliója</b><br><br>"
                    "Kockázat (CVaR): %{x:.2%}<br>"
                    "Várható Hozam: %{y:.2%}%{customdata}<extra></extra>"
                ),
            )
        )

    # A separate compact legend in the empty upper-left corner maps each
    # bootstrap cloud to its marker shape. The opaque dummy swatches are also
    # clickable: each one toggles its own cloud without hiding the others.
    for rank, key in enumerate(SIMULATION_LABELS, start=1):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=COLORS[key],
                    size=9,
                    symbol=SIMULATION_SYMBOLS[key],
                    line=dict(color=COLORS[key], width=0.8),
                ),
                name=SIMULATION_LABELS[key],
                legend="legend2",
                legendgroup=f"bootstrap_{key}",
                legendrank=rank,
                hoverinfo="skip",
            )
        )

figure.add_trace(
    go.Scatter(
        x=base_frontier["cvar"],
        y=base_frontier["ret"],
        mode="lines",
        line=dict(color=COLORS["frontier"], width=3),
        name="Hatékony front",
        legendrank=1,
    )
)

asset_cvar = base_assets_df["cvar"].to_numpy()
for index, asset_name in enumerate(asset_classes):
    asset_label = re.sub(r"\s*\([^)]*\)", "", asset_name)
    asset_weights = identity[index]
    weights_text_asset = format_weights_for_tooltip(asset_weights, asset_classes)
    cvar = float(asset_cvar[index])
    # Put labels on the inside of the plot so the rightmost asset cannot be
    # clipped at the chart edge. The lowest-risk asset sits on the y-axis and
    # next to the Min Risk point, so its label goes below the marker; the
    # highest-risk asset gets a label above the marker.
    if index == int(np.argmin(asset_cvar)):
        text_position = "bottom right"
    elif index == int(np.argmax(asset_cvar)):
        text_position = "top center"
    elif cvar > base_assets_df["cvar"].median():
        text_position = "middle left"
    else:
        text_position = "middle right"
    figure.add_trace(
        go.Scatter(
            x=[cvar],
            y=[base_assets_df["ret"].iloc[index]],
            mode="markers+text",
            marker=dict(
                size=12,
                color=COLORS["assets"],
                symbol="diamond",
                line=dict(color="#555555", width=1),
            ),
            text=asset_label,
            textposition=text_position,
            cliponaxis=False,
            name="Egyedi eszközök",
            legendgroup="assets",
            legendrank=2,
            showlegend=(index == 0),
            customdata=[weights_text_asset],
            hovertemplate=(
                f"<b>{asset_name}</b><br><br>"
                "Kockázat (CVaR): %{x:.2%}<br>"
                "Várható Hozam: %{y:.2%}%{customdata}<extra></extra>"
            ),
        )
    )

symbols = {"min_risk": "circle", "tangency": "square", "risk_parity": "triangle-up"}
for rank, (key, label) in enumerate(PORTFOLIO_NAMES.items(), start=3):
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
                line=dict(width=1, color="#555555"),
            ),
            name=label,
            legendrank=rank,
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
            line=dict(width=1, color="#555555"),
        ),
        name="Ön portfóliója",
        legendrank=6,
        hovertemplate=(
            "<b>Az Ön portfóliója</b><br><br>"
            "Kockázat (CVaR): %{x:.2%}<br>"
            "Várható Hozam: %{y:.2%}%{text}<extra></extra>"
        ),
    )
)

figure.update_layout(
    xaxis=dict(
        title=f"Havi CVaR ({confidence:.0%}) – kockázat",
        tickformat=".1%",
        hoverformat=".2%",
        nticks=6,
        automargin=True,
        zeroline=False,
    ),
    yaxis=dict(
        title="Évesített várható hozam",
        tickformat=".1%",
        hoverformat=".2%",
        nticks=6,
        automargin=True,
        zeroline=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5,
        entrywidth=110,
        entrywidthmode="pixels",
        groupclick="togglegroup",
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    legend2=dict(
        title=dict(text="Bootstrap", font=dict(size=10)),
        orientation="v",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        groupclick="togglegroup",
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0)",
        itemsizing="constant",
    ),
    height=540,
    margin=dict(l=10, r=10, t=10, b=120),
    hovermode="closest",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(
    figure,
    width="stretch",
    alt=(
        "Szórásdiagram a portfóliók havi CVaR kockázata és évesített várható "
        "hozama között. A hatékony front mellett a Min Risk, Tangency, Risk "
        "Parity és az Ön portfóliója, valamint az egyedi eszközök láthatók."
    ),
)

min_risk_index = int(np.argmax(base_special_weights["min_risk"]))
min_risk_note = ""
if base_special_weights["min_risk"][min_risk_index] >= 0.995:
    min_risk_note = (
        f" A Min Risk portfólió {ticker_label(asset_classes[min_risk_index])} "
        "eszközben van 100%-ban."
    )
st.caption(
    f"{frontier_insight(user_cvar, user_ret, base_frontier)} "
    "A Risk Parity portfólió szórásalapú kockázatparitás; a CVaR a chart "
    f"kockázati tengelye.{min_risk_note}"
)

# --- Detailed current portfolio metrics ------------------------------------
st.subheader("Az Ön portfóliója")
metric_columns = st.columns(2)
metric_columns[0].metric(
    "Évesített várható hozam",
    f"{user_ret:.2%}",
    help="A havi átlagos hozam ×12; ez nem hozam-előrejelzés.",
)
metric_columns[1].metric(
    f"Havi CVaR ({confidence:.0%})",
    f"{user_cvar:.2%}",
    help=(
        f"A legrosszabb {1 - confidence:.0%} havi hozam átlagos vesztesége. "
        "A CVaR havi, a hozam évesített mértékegységű."
    ),
)
st.caption(
    "A hozam évesített, a CVaR havi veszteség. A két mutató eltérő "
    "időhorizonton értelmezendő."
)

# --- Weights and metrics tables --------------------------------------------
weights_by_portfolio = {**base_special_weights, "user": weights_vector}
asset_legend = " · ".join(
    f"{ticker_label(asset)} = {re.sub(r'\s*\([^)]*\)', '', asset)}"
    for asset in asset_classes
)
with st.expander("Portfóliósúlyok és mutatók"):
    weights_tab, metrics_tab = st.tabs(["Súlyok", "Mutatók"])
    with weights_tab:
        weights_table = build_weights_table(asset_classes, weights_by_portfolio)
        st.dataframe(
            weights_table.style.format(
                {
                    column: "{:.2%}"
                    for column in weights_table.columns
                    if column != "Portfólió"
                }
            ),
            width="stretch",
            height="auto",
            hide_index=True,
            alt="Portfóliósúlyok: soronként egy portfólió, oszloponként egy ticker.",
        )
        st.caption(asset_legend)

    with metrics_tab:
        metrics_table = build_metrics_table(
            asset_classes,
            weights_by_portfolio,
            base_er,
            returns_data,
            confidence,
        )
        st.dataframe(
            metrics_table.style.format(
                {
                    "Évesített hozam": "{:.2%}",
                    "Havi CVaR": "{:.2%}",
                }
            ),
            width="stretch",
            height="auto",
            hide_index=True,
            alt=(
                "Portfóliónkénti évesített hozam és havi CVaR. "
                "A CVaR havi veszteség, a hozam évesített."
            ),
        )
        st.caption(
            "A hozam évesített, a CVaR havi veszteség; a két oszlop eltérő "
            "időhorizontú."
        )

# --- Methodology ------------------------------------------------------------
with st.expander("Módszertan és korlátok"):
    st.markdown(
        f"""
        - **Adatok:** havi hozamok {returns_data.index.min():%Y-%m} és
          {returns_data.index.max():%Y-%m} között, {len(asset_classes)} ETF-ből.
        - **Minta-bizonytalanság:** *joint bootstrap*. Minden szimulációban
          újramintavételezzük a havi hozamokat, majd ugyanabból a mintából
          számítjuk a várható hozamot, a CVaR-t és a hatékony frontot.
        - **Várható hozam:** a bootstrap minta átlaga × 12 (évesítve).
        - **Kockázat:** havi empirikus CVaR (Expected Shortfall),
          {confidence:.0%}-os konfidenciaszinten. A hozam évesített, a CVaR
          havi mértékegységű.
        - **Risk parity:** a variancia-hozzájárulásokat egyenlíti ki
          (szórásalapú, mind a négy eszközre). A CVaR-alapú
          hozzájárulások a farokban negatívak lehetnek a cash esetében,
          ezért a CVaR-risk-paritás gyakran sarokmegoldást ad (BIL nélkül),
          és instabil a bootstrap alatt. A charton a Risk Parity pont
          CVaR-je a szokásos módon számolódik.
        - **Megkötések:** hosszú pozíciók, teljes befektetés, tranzakciós
          költség nélkül.
        - **Korlátok:** i.i.d. bootstrap (nincs idősor-függőség), in-sample
          becslés, konstans kockázatmentes hozam, kis eszközuniverzum.
        """
    )
