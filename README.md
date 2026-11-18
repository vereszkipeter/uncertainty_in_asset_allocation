# uncertainty_in_asset_allocation

A small research dashboard for studying how uncertainty in **input estimates**
(expected returns and the return distribution) propagates into a CVaR-based
asset-allocation decision.

The app visualizes the point-estimate efficient frontier, three reference
portfolios (minimum risk, CVaR tangency and volatility risk parity), and the
bootstrap uncertainty cloud around them.

## What is in the repository

| Path | Purpose |
| --- | --- |
| `src/prepare_data.py` | Downloads/caches monthly returns, runs the joint bootstrap, and writes the HDF5 bundle. |
| `src/app.py` | Streamlit dashboard (Hungarian UI). |
| `src/portfolio.py` | Shared portfolio math: CVaR, variance/CVaR contributions, optimizers, frontier, bootstrap helpers. |
| `src/data_io.py` | Versioned HDF5 persistence. |
| `src/config.py` | Asset universe, paths, seeds, and simulation defaults. |
| `data/monthly_returns.csv` | Cached monthly simple returns, 2010-01 to 2024-12. |
| `data/asset_allocation_uncertainty.h5` | Pre-computed simulation bundle used by the app. |
| `tests/` | Pytest suite for the math, pipeline, and HDF5 round-trip. |

## Assets and data

The universe contains four USD-denominated ETFs:

- `USD Készpénz (BIL)` — 1-3 month Treasury bill ETF (cash proxy)
- `USA Állampapír (IEF)` — 7-10 year Treasury bond ETF
- `USA Váll. Kötvény (LQD)` — investment-grade corporate bond ETF
- `Arany (GLD)` — gold ETF

Monthly simple returns are computed from Yahoo Finance adjusted close prices
with `resample("ME").last().pct_change()`. The CSV cache is used by default so
the committed results are reproducible without network access. Pass
`--force-download` to refresh the cache.

## Methodology

### Point estimates

- Expected returns are annualized as `12 * monthly mean`.
- Risk is the empirical **CVaR (Expected Shortfall)** at the 95% level on
  monthly returns, reported as a positive loss number.
- Portfolios are long-only and fully invested. The optimizers use SLSQP with
  weights in `[0, 1]` and a sum-to-one constraint.
- The efficient frontier minimizes CVaR for a grid of target annualized
  returns. The grid starts at the minimum-risk portfolio's return, so
  dominated, backward-bending points are not plotted. Any remaining dominated
  points found by the numerical optimizer are removed by a lower-envelope
  filter.

### Risk parity

Risk parity is defined on **portfolio variance**, not on CVaR. The weights
solve the convex log-barrier problem

```
minimize 0.5 * w' Sigma w - (1 / N) * sum(log(w_i)),  w_i > 0,
```

whose first-order condition is `w_i * (Sigma w)_i = 1 / N`: every asset
contributes the same share of portfolio variance. The solution is unique for
a positive-definite covariance matrix and gives every asset a strictly
positive weight. The raw solution is normalized to sum to one.

CVaR risk parity was tried first and rejected for this asset universe. The
Euler CVaR contribution of cash can be negative because cash tends to hedge
the portfolio's worst months. Equal positive CVaR contributions are then
infeasible, so the optimizer returns a corner solution: the point-estimate
portfolio held 0% cash and split the CVaR budget equally across IEF, LQD and
GLD, with an objective value of 0.083 instead of 0. Across bootstrap samples
the cash weight jumped between 0% and 98%, and some stored solutions were
local minima. Volatility risk parity avoids both problems: the point estimate
is 88.3% BIL, 5.05% IEF, 4.36% LQD and 2.29% GLD, with 25% variance
contributions, and the bootstrap weights are stable.

The chart's risk axis remains CVaR, so the volatility risk-parity point is
not necessarily CVaR-efficient. It is plotted as a reference portfolio, not
as a frontier solution.

### Input uncertainty: joint bootstrap

The previous implementation drew expected returns from a normal distribution
whose covariance was scaled by `12` instead of `144`. Because the annualized
mean is `12 * monthly mean`, its sampling covariance is

```
Cov(12 * mean) = 144 * Sigma / n,
```

not `12 * diag(Sigma) / n`. The old code therefore understated the standard
deviation of the expected-return estimate by a factor of `sqrt(12) ≈ 3.46`.

The pipeline now uses a **joint row bootstrap** instead:

1. Resample the monthly return rows with replacement.
2. Use the bootstrap sample mean × 12 as the expected-return draw.
3. Use the **same** bootstrap sample to estimate CVaR and the efficient
   frontier.

This keeps expected-return uncertainty and distributional uncertainty
coherent, captures the full covariance structure, and avoids a separate
normality assumption for the mean estimator.

### Reproducibility

The default bundle uses `seed=42` and 100 simulations. The seed and simulation
count are stored as HDF5 attributes and shown in the app caption. Re-running
`prepare_data.py` with the same seed reproduces the bundle; the test suite
checks this explicitly.

## Installation

Python 3.12 is recommended. The versions below are the ones used for the
committed bundle.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Build the data bundle

```bash
python src/prepare_data.py
```

Useful options:

```bash
python src/prepare_data.py --simulations 200 --seed 7
python src/prepare_data.py --force-download
python src/prepare_data.py --start 2015-01-01 --end 2024-12-31
```

Dates passed to `--start` and `--end` are inclusive. Run
`python src/prepare_data.py --help` for all options.

## Run the app

```bash
streamlit run src/app.py
```

The app reads `data/asset_allocation_uncertainty.h5`. If the file is missing or
was written with an older schema, the app shows an error asking you to run
`python src/prepare_data.py`.

### Dashboard layout and visual conventions

- The page title is deliberately short; the longer title is kept as HDF5
  metadata. The chart has no duplicate in-chart title.
- The simulation control is the first sidebar control and is labelled
  **Minta bizonytalanság** (sample uncertainty), because the bootstrap
  resamples the historical return sample rather than a structural model.
- The chart legend sits below the plot and uses the Okabe-Ito
  colorblind-safe palette. Bootstrap traces share one opaque legend entry
  (`Bootstrap felhők`); the underlying clouds are thin, translucent outlines
  with distinct marker shapes per portfolio.
- The current portfolio's annualized return and monthly CVaR are shown
  together with help text clarifying their different horizons.
- The chart annotation quantifies the CVaR gap between the user's portfolio
  and the efficient frontier at the same expected return, and flags a
  min-risk portfolio that is 100% cash.
- Weights are shown as a transposed table (portfolios in rows, tickers in
  columns) and a second table reports annualized return and monthly CVaR.
  Both live in the **Portfóliósúlyok és mutatók** expander. The chart also
  has an `alt` description for screen readers.

## Tests

```bash
python -m pytest
```

The suite covers:

- CVaR, variance risk contributions, and weight validation;
- volatility risk-parity positivity, equal contributions, and bootstrap stability;
- feasibility and optimality of the three reference portfolios;
- monotonicity of the efficient frontier;
- the `144 * Sigma / n` covariance scaling and the joint-bootstrap covariance;
- pipeline reproducibility and shape/consistency checks;
- HDF5 round-trip and schema-version rejection;
- the dashboard's table builders, ticker labels, frontier interpolation and
  insight text;
- a Streamlit `AppTest` smoke test for the page title, sidebar controls,
  tables and the simulation toggle.

## Validation of the committed bundle

With the committed data (`n=180`, `100` simulations, `seed=42`):

- The equal-weight portfolio's empirical bootstrap variance of the
  annualized expected return is within **1.5%** of the theoretical
  `(n-1)/n * 144 * Sigma / n` reference.
- The diagonal standard-deviation ratios are between **0.89 and 1.10** of the
  theoretical bootstrap reference (sampling noise with 100 draws).
- All 100 special-portfolio optimizations converged; the volatility
  risk-parity solve is convex and all weights are strictly positive.
- The volatility risk-parity point estimate is 88.3% BIL, 5.05% IEF,
  4.36% LQD and 2.29% GLD, with variance contribution shares of 25% each.
  Across the 100 bootstrap samples its weights are stable: BIL
  88.2% ± 0.9%, IEF 5.1% ± 0.5%, LQD 4.4% ± 0.4% and GLD 2.3% ± 0.2%.
  The tangency weights remain highly dispersed, as expected.
- All 100 efficient frontiers are non-empty and monotone in both CVaR and
  expected return; the average number of retained points is 19.5 out of 20.
- The full test suite passes: **32 tests**.

These are statistical checks, not a guarantee that the point estimates are
unbiased. The bootstrap is conditional on the historical sample.

## Limitations

- The bootstrap is i.i.d. across months and ignores serial dependence,
  volatility clustering, and regime shifts.
- Estimation is in-sample; there is no out-of-sample or transaction-cost
  analysis.
- The risk-free rate is a constant 2% assumption, while the asset universe
  includes a cash proxy (BIL).
- CVaR at 95% with 180 monthly observations uses roughly nine tail
  observations, so the risk estimate itself is noisy.
- Risk parity is volatility-based, while the chart's risk axis is CVaR; the
  risk-parity point is therefore not necessarily CVaR-efficient. CVaR risk
  parity was rejected because cash can have a negative CVaR contribution,
  producing unstable corner solutions.
- The asset universe is small and defensive; the minimum-risk portfolio can
  legitimately concentrate in cash.
- Expected returns are annualized by ×12 rather than by compounding; this is a
  deliberate convention for the risk-return plot, not a forecast.

## HDF5 schema (version 2)

```
attrs: title, schema_version, seed, n_simulations,
       confidence_level, risk_free_rate
base/
  returns        [n_obs, n_assets] float64
  dates          [n_obs] ISO-8601 strings
  asset_names    [n_assets] UTF-8 strings
simulation_inputs/
  expected_returns   [n_sim, n_assets] float64
  bootstrap_returns  [n_sim, n_obs, n_assets] float64
simulation_outputs/
  min_risk_points / min_risk_weights
  tangency_points / tangency_weights
  risk_parity_points / risk_parity_weights
  efficient_frontiers/frontier_{i}  [n_points, 2] (cvar, ret)
```

The `risk_parity_*` arrays contain **volatility** risk-parity weights (equal
variance contributions), not CVaR risk-parity weights. Bootstrap samples are
stored as a 3-D array; their row order is the resampled order and carries no
date meaning. Dates are stored as ISO strings to avoid
platform-dependent integer units.
