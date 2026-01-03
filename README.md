# Fed Action & Fed Communication — Event Regression Analysis

## Overview
This repository reproduces an empirical analysis that studies how U.S. monetary policy actions and FOMC communications affect FX forward and spot rates (examples: USD/VND and USD/CNY, one‑year forward). The analysis is implemented as Jupyter notebooks that use utility modules in `src/utils/` for data preparation and for running event-based rolling regressions.

## Project structure
```
LICENSE
README.md
requirements.txt
data/                 # raw data used by notebooks
src/
  event_regression.ipynb
  robustness.ipynb
  utils/
    __init__.py
    data_merger.py
    event_rolling.py
```

## Requirements
- Python 3.9 or later
- Install dependencies: `pip install -r requirements.txt`
- Recommended: use a virtual environment (venv or conda)

## Recommended workflow (important)
The notebooks import the `utils` package using a relative top-level import (e.g. `from utils import load_data`). To ensure imports work correctly, either:

Option A (recommended): start Jupyter from the `src/` folder so `src/` is the notebook working directory:

```powershell
cd src
jupyter lab    # or jupyter notebook
```

Option B: start Jupyter from repo root and add `src/` to `PYTHONPATH` or modify `sys.path` at the top of the notebook:

```python
import sys
sys.path.append('src')
from utils import load_data, EventRolling, EventRollingConfig
```

## Notebooks
- `src/event_regression.ipynb` — primary notebook that runs the event regression pipeline, produces summary tables and diagnostic figures.
- `src/robustness.ipynb` — robustness checks: alternative windows, event definitions, and additional visualizations.

Open the notebook(s) and run cells top-to-bottom to reproduce results.

## Quick install and run (Windows example)
1. Clone the repository.
2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. Start Jupyter from `src/` (recommended) and open the notebooks:

```powershell
cd src
jupyter lab
```

4. Code overview (what each component does)

- `src/event_regression.ipynb` — Primary notebook:
  - Loads and preprocesses data using `load_data()` from `src/utils/data_merger.py`.
  - Configures and runs event-based regressions using `EventRolling` and `EventRollingConfig` from `src/utils/event_rolling.py`.
  - Aggregates results with `summarize_results()` and generates diagnostic tables and figures.
  - Typical outputs: time series of β coefficients, per-event R², summary tables of statistically significant events, and diagnostic plots.

- `src/robustness.ipynb` — Robustness checks:
  - Tests alternative event windows (±10, ±15, ±21 days), statement-day-only events, and minutes-only events.
  - Compares sensitivity of results to event definitions and window sizes.

- `src/utils/data_merger.py` — Data loading and preprocessing:
  - `load_data()` merges Fed funds, FOMC, forward and spot data, standardizes column names, and generates features (log levels, differences, event dummies), returning a ready-to-use DataFrame.

- `src/utils/event_rolling.py` — Event regression utilities:
  - `EventRollingConfig`: configuration for days_before/days_after, lags, significance level, covariance type, etc.
  - `EventRolling`: runs regressions (e.g., `rolling_event_regression_asymmetric()`), summarizes results (`summarize_results()`), and provides plotting functions for event quality.

5. Example run (programmatic usage)

```python
# Run from src/ (or ensure src/ is in PYTHONPATH)
from utils import load_data, EventRolling, EventRollingConfig

# Load
df = load_data()

# Configure and run
cfg = EventRollingConfig(days_before=21, days_after=21, event_col='fomc_event')
er = EventRolling(df=df, x_lagged_vars=['fed_funds'], x_non_lagged_vars=['fomc_Monetary Policy Stance'], y_vars=['Bid_FW_usd_vnd_1y'], config=cfg)
results = er.rolling_event_regression_asymmetric()
summary = er.summarize_results(results)
```

For reproducibility, please ensure required input data are available in the `data/` directory.
