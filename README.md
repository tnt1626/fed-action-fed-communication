# Fed Action & Fed Communication — Event Regression Analysis

## Overview
This repository contains the code and data needed to reproduce an empirical analysis that studies how U.S. monetary policy actions and FOMC communications affect FX forward and spot rates (examples: USD/VND and USD/CNY, one‑year forward). The analysis is implemented in a Jupyter notebook (`src/event_regression.ipynb`) and relies on utility modules in `src/utils/` for data preparation and event-based rolling regressions.

## Requirements
- Python 3.9 or later
- Install Python dependencies: `pip install -r requirements.txt`
- Recommended: use a virtual environment (venv or conda)

## Quick install and run
1. Clone the repository.
2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Create an `outputs/` directory to collect figures and tables:

```powershell
mkdir outputs
```

4. Open `src/event_regression.ipynb` in JupyterLab or VS Code and run the notebook from top to bottom to reproduce the analysis and generate outputs.

## Programmatic usage (example)
You can also use the modules directly from Python scripts or a REPL. Example:

```python
from src.utils.data_merger import load_data
from src.utils.event_rolling import EventRolling, EventRollingConfig

# Load prepared dataset
df = load_data()

# Configure and run event rolling regressions
config = EventRollingConfig(window_size=21, days_before=21, days_after=21, lags=[0])
er = EventRolling(
    df=df,
    x_lagged_vars=['delta_log_Bid_FW_usd_vnd_2w'],
    x_non_lagged_vars=['fed_funds'],
    y_vars=['delta_log_Bid_FW_usd_vnd_2w'],
    config=config
)
results = er.rolling_event_regression_asymmetric()
summary = er.summarize_results(results)
er.plot_event_quality(results)
```

Adjust the variable names in the example to match columns created by `load_data()` in your local dataset.

## Code reference — key modules and functions
This section briefly documents the main functions and classes you will use.

### src/utils/data_merger.py
- load_data() -> pd.DataFrame
  - Loads and merges the analysis inputs including Federal Funds Rate, FOMC data, forward and spot rates. Produces cleaned, transformed columns (log levels and log-differences) required by the analysis pipeline.

- load_fed_funds_rate_data() -> pd.DataFrame
  - Reads the Federal Funds Rate file and standardizes column names and date formatting.

- load_fomc_data(df: pd.DataFrame) -> pd.DataFrame
  - Reads FOMC decision data, encodes qualitative decisions into numeric dummies (decrease=1, maintain=2, increase=3), and merges into `df` by date.

- process_fw_spot_data(df: pd.DataFrame, filename: str, rate_type: Literal['FW','Spot'], skip_rows: int) -> pd.DataFrame
  - Imports forward/spot sheets, renames columns, merges them into `df`, drops rows missing required rate columns, and computes `log_` and `delta_log_` features.

### src/utils/event_rolling.py
- EventRollingConfig
  - Dataclass holding configuration for rolling windows, lag structure, covariance type, significance level, and plotting options.

- EventRolling
  - __init__(df, x_lagged_vars, x_non_lagged_vars, y_vars, config)
    - Prepares lagged/lead variables and identifies event indices using `event_col`.

  - rolling_event_regression() -> pd.DataFrame
    - Runs symmetric rolling regressions across events using `window_size`.

  - rolling_event_regression_asymmetric() -> pd.DataFrame
    - Runs asymmetric (days_before / days_after) rolling regressions per event.

  - summarize_results(results_df: pd.DataFrame) -> pd.DataFrame
    - Aggregates and summarizes coefficient significance, counts, and extreme effects across events.

  - plot_event_quality(results_df: pd.DataFrame, r2_threshold: float = 0.05) -> None
    - Produces a time series plot of event R² and highlights events that pass quality criteria.


## Reproducibility notes
- Use `requirements.txt` to replicate the Python environment. For full reproducibility, generate and commit an environment export (e.g., `pip freeze > pinned-requirements.txt` or a `conda` `environment.yml`).
- Where randomness matters, set explicit seeds within the notebook prior to running stochastic steps.
