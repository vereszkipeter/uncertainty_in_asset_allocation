# prepare_data.py (Javított, formázott és kommentezett verzió)

import pandas as pd
import numpy as np
import yfinance
from scipy.optimize import minimize
import h5py
import warnings
import time
import os

# A scipy optimalizáció során előforduló futásidejű figyelmeztetések elrejtése.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- KONFIGURÁCIÓS VÁLTOZÓK ---
RISK_FREE_RATE = 0.02  # Kockázatmentes hozam (éves)
N_SIMULATIONS = 100  # A Monte Carlo szimulációk száma
START_DATE = "2010-01-01"  # Adatletöltés kezdő dátuma
END_DATE = "2024-12-31"  # Adatletöltés záró dátuma
DATA_DIR = "data"  # Könyvtár az adatok tárolására
OUTPUT_FILE = os.path.join(DATA_DIR, "asset_allocation_uncertainty.h5")  # Kimeneti HDF5 fájl
RETURNS_CSV_FILE = os.path.join(DATA_DIR, "monthly_returns.csv")  # Hozamok cache fájlja

# Eszközosztályok és a hozzájuk tartozó Yahoo Finance tickerek.
TICKERS = {
    "USD Készpénz (BIL)": "BIL",
    "USA Állampapír (IEF)": "IEF",
    "USA Váll. Kötvény (LQD)": "LQD",
    "Arany (GLD)": "GLD",
}
ASSET_CLASSES = list(TICKERS.keys())
TICKER_LIST = list(TICKERS.values())


# --- ADATLETÖLTŐ ÉS CACHELŐ FÜGGVÉNY ---
def get_returns_data(csv_path, tickers, start_date, end_date):
    """
    Letölti vagy a helyi CSV cache-ből betölti az eszközök havi hozamait.
    A letöltött adatokat elmenti a gyorsabb későbbi betöltés érdekében.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        print(f"Hozamadatok betöltése a helyi cache-ből: '{csv_path}'")
        return pd.read_csv(csv_path, index_col="date", parse_dates=True)

    print("Helyi cache nem található. Adatok letöltése a yfinance segítségével...")
    try:
        # Adatok letöltése és 'Adj Close' helyett 'Close' használata auto_adjust=True-val
        full_data = yfinance.download(
            tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
        )
        # Havi hozamok számítása: hónap végi ('M') adatokkal és százalékos változással.
        returns_df = full_data["Close"].resample("M").ffill().pct_change().dropna()

        # Adatstruktúra egységesítése
        if isinstance(returns_df, pd.Series):
            returns_df = returns_df.to_frame(name=ASSET_CLASSES[0])
        else:
            returns_df.columns = ASSET_CLASSES
        returns_df.index.name = "date"
        returns_df.to_csv(csv_path)
        print(f"Adatok sikeresen lementve a cache-be: '{csv_path}'")
        return returns_df
    except Exception as e:
        print(f"Hiba az adatletöltés során: {e}")
        return pd.DataFrame()


# --- PORTFÓLIÓ-OPTIMALIZÁCIÓS FÜGGVÉNYEK ---
def calculate_portfolio_cvar(weights, monthly_returns):
    """Kiszámítja a portfólió Feltételes Várható Értékét (CVaR) 95%-os konfidenciaszinten."""
    portfolio_returns = monthly_returns.dot(np.array(weights))
    var = np.quantile(portfolio_returns, 0.05)  # 5%-os Value at Risk (VaR)
    # A CVaR a VaR-nál rosszabb hozamok átlaga. Negatív előjellel adjuk vissza, mert a kockázatot minimalizáljuk.
    return -portfolio_returns[portfolio_returns <= var].mean()


def calculate_portfolio_return(weights, expected_returns):
    """Kiszámítja a portfólió súlyozott évesített várható hozamát."""
    return np.sum(np.array(weights) * expected_returns)


def get_portfolio_metrics(weights, expected_returns, monthly_returns):
    """Egyetlen függvényben adja vissza a portfólió CVaR és várható hozam metrikáit."""
    cvar = calculate_portfolio_cvar(weights, monthly_returns)
    ret = calculate_portfolio_return(weights, expected_returns)
    return cvar, ret


def neg_sharpe_ratio_cvar(weights, expected_returns, monthly_returns, risk_free_rate):
    """A CVaR-alapú Sharpe-ráta negatívját számolja. Optimalizációhoz használjuk (minimalizálás)."""
    p_cvar, p_ret = get_portfolio_metrics(weights, expected_returns, monthly_returns)
    # Kezeljük az esetet, ha a kockázat (CVaR) nulla vagy negatív.
    return -(p_ret - risk_free_rate) / p_cvar if p_cvar > 1e-6 else np.inf


def risk_parity_cvar_objective(weights, monthly_returns):
    """
    CVaR-alapú kockázatparitás célfüggvény.
    A cél, hogy minden eszköz hozzájárulása a portfólió farokkockázatához (CVaR) egyenlő legyen.
    """
    weights, num_assets = np.array(weights), len(weights)
    portfolio_returns = monthly_returns.dot(weights)
    var_threshold = np.quantile(portfolio_returns, 0.05)
    tail_scenarios = monthly_returns[portfolio_returns <= var_threshold]

    if tail_scenarios.empty:
        return 1e6  # Ha nincsenek farokesemények, nagy büntetést adunk vissza.

    # Kockázati hozzájárulások: súly * az eszköz átlagos negatív hozama a farokesemények során.
    risk_contributions = weights * (-tail_scenarios.mean(axis=0).values)
    total_risk = np.sum(risk_contributions)

    if total_risk <= 1e-6:
        return 1e6

    # A cél a relatív kockázati hozzájárulások és az egyenlő (1/N) arány közötti négyzetes eltérés minimalizálása.
    return np.sum(((risk_contributions / total_risk) - (1 / num_assets)) ** 2)


def calculate_special_portfolios(expected_returns, monthly_returns, risk_free_rate):
    """Kiszámítja a három nevezetes portfólió (Minimum Kockázat, Érintő, Kockázatparitás) súlyait."""
    num_assets = len(expected_returns)
    # Kezdeti súlyok: egyenlő elosztás.
    initial_weights = np.full(num_assets, 1.0 / num_assets)
    # Korlátok: minden súly 0 és 1 között legyen (nincs shortolás).
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    # Megszorítás: a súlyok összege legyen 1.
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Minimum Kockázatú portfólió (CVaR minimalizálása)
    min_risk_res = minimize(
        calculate_portfolio_cvar, initial_weights, args=(monthly_returns,),
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    # Érintő portfólió (Sharpe-ráta maximalizálása)
    tangency_res = minimize(
        neg_sharpe_ratio_cvar, initial_weights, args=(expected_returns, monthly_returns, risk_free_rate),
        method="SLSQP", bounds=bounds, constraints=constraints
    )
    # Kockázatparitásos portfólió
    risk_parity_res = minimize(
        risk_parity_cvar_objective, initial_weights, args=(monthly_returns,),
        method="SLSQP", bounds=bounds, constraints=constraints
    )

    return {
        "min_risk": min_risk_res.x,
        "tangency": tangency_res.x,
        "risk_parity": risk_parity_res.x,
    }


def calculate_efficient_frontier(expected_returns, monthly_returns, n_points=20):
    """Kiszámítja a hatékony front pontjait a CVaR-hozam térben."""
    results = []
    # Célhozamok definiálása a lehetséges hozamtartományban.
    target_returns = np.linspace(min(expected_returns), max(expected_returns), n_points)
    bounds = tuple((0.0, 1.0) for _ in range(len(expected_returns)))
    initial_weights = np.full(len(expected_returns), 1.0 / len(expected_returns))

    for target_ret in target_returns:
        # Megszorítások: súlyok összege 1, és a portfólió hozama egyenlő a célhozammal.
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: calculate_portfolio_return(w, expected_returns) - target_ret},
        )
        # Optimalizálás: CVaR minimalizálása adott hozamszint mellett.
        res = minimize(
            calculate_portfolio_cvar, initial_weights, args=(monthly_returns,),
            method="SLSQP", bounds=bounds, constraints=constraints
        )
        if res.success:
            results.append({"cvar": res.fun, "ret": target_ret})

    return (
        pd.DataFrame(results).sort_values(by="cvar").drop_duplicates()
        if results
        else pd.DataFrame(columns=["cvar", "ret"])
    )


# --- FŐ VÉGREHAJTÓ BLOKK ---
if __name__ == "__main__":
    start_time = time.time()
    returns_data = get_returns_data(RETURNS_CSV_FILE, TICKER_LIST, START_DATE, END_DATE)
    if returns_data.empty:
        print("Hiba: A hozamadatok üresek. A program leáll.")
        exit()

    print(f"\nIndul a {N_SIMULATIONS} db Monte Carlo szimuláció...")
    num_obs, num_assets = returns_data.shape

    # Bemeneti paraméterek a szimulációhoz:
    # - Várható hozamok (évesítve) a historikus átlagokból.
    base_er = returns_data.mean().values * 12
    # - A várható hozamok becslési hibájának kovarianciamátrixa.
    er_cov = np.diag(((returns_data.std() / np.sqrt(num_obs)) * np.sqrt(12)) ** 2)

    # Tárolók a szimulációk inputjainak és outputjainak.
    sim_inputs = {"expected_returns": [], "bootstrap_samples": []}
    sim_outputs = {
        "frontiers": [],
        "min_risk_points": [], "tangency_points": [], "risk_parity_points": [],
        "min_risk_weights": [], "tangency_weights": [], "risk_parity_weights": [],
    }

    # A Monte Carlo ciklus
    for i in range(N_SIMULATIONS):
        print(f"  - Feldolgozás: {i + 1}/{N_SIMULATIONS}")
        # 1. Várható hozamok szimulálása (becslési bizonytalanság).
        sim_er = np.random.multivariate_normal(mean=base_er, cov=er_cov)
        # 2. Hozam-idősor szimulálása bootstrap módszerrel (eloszlás bizonytalansága).
        bootstrap_sample = returns_data.sample(n=num_obs, replace=True)

        # Inputok mentése
        sim_inputs["expected_returns"].append(sim_er)
        sim_inputs["bootstrap_samples"].append(bootstrap_sample)

        # Számítások és outputok mentése az adott szimulációra
        sim_outputs["frontiers"].append(
            calculate_efficient_frontier(sim_er, bootstrap_sample)
        )
        special_weights = calculate_special_portfolios(
            sim_er, bootstrap_sample, RISK_FREE_RATE
        )

        for key in ["min_risk", "tangency", "risk_parity"]:
            metrics = get_portfolio_metrics(special_weights[key], sim_er, bootstrap_sample)
            sim_outputs[f"{key}_points"].append(metrics)
            sim_outputs[f"{key}_weights"].append(special_weights[key])

    print("\nSzimulációk sikeresen lefutottak.")
    print(f"Eredmények mentése a(z) '{OUTPUT_FILE}' fájlba...")

    # Eredmények mentése HDF5 fájlba
    with h5py.File(OUTPUT_FILE, "w") as f:
        f.attrs["title"] = "Az eszközallokáció input adataiban rejlő bizonytalanság"
        
        # Alap adatok (historikus)
        base_group = f.create_group("base")
        base_group.create_dataset("returns", data=returns_data.to_numpy())
        base_group.create_dataset("dates", data=returns_data.index.astype(np.int64))
        # JAVÍTÁS: A string listát közvetlenül adjuk át, elkerülve a dtype='S' hibát.
        base_group.create_dataset("asset_names", data=ASSET_CLASSES)
        
        # Szimulációs inputok
        inputs_group = f.create_group("simulation_inputs")
        inputs_group.create_dataset("expected_returns", data=np.array(sim_inputs["expected_returns"]))
        
        # A bootstrap minták egyetlen nagy táblázatként mentése a hatékonyságért.
        all_samples_df = pd.concat(
            sim_inputs["bootstrap_samples"], keys=range(N_SIMULATIONS), names=["sim_id"]
        ).reset_index()
        date_column_name = all_samples_df.columns[1]
        all_samples_df[date_column_name] = all_samples_df[date_column_name].astype(np.int64)
        all_samples_df.rename(columns={date_column_name: "date"}, inplace=True)
        inputs_group.create_dataset("bootstrap_samples", data=all_samples_df.to_numpy())
        # JAVÍTÁS: Az oszlopneveket tartalmazó string listát is közvetlenül adjuk át.
        inputs_group.create_dataset(
            "bootstrap_samples_columns", data=all_samples_df.columns.tolist()
        )

        # Szimulációs outputok
        outputs_group = f.create_group("simulation_outputs")
        for key in ["min_risk", "tangency", "risk_parity"]:
            outputs_group.create_dataset(f"{key}_points", data=np.array(sim_outputs[f"{key}_points"]))
            outputs_group.create_dataset(f"{key}_weights", data=np.array(sim_outputs[f"{key}_weights"]))

        # A szimulált hatékony frontok külön csoportba kerülnek.
        frontiers_group = outputs_group.create_group("efficient_frontiers")
        for i, df in enumerate(sim_outputs["frontiers"]):
            if not df.empty:
                frontiers_group.create_dataset(f"frontier_{i}", data=df.to_numpy())

    end_time = time.time()
    print(
        f"\nMinden adat sikeresen elmentve. Futási idő: {end_time - start_time:.2f} másodperc."
    )