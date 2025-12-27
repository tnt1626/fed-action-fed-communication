# Fed Action & Fed Communication — Event Regression Analysis

Short description
-----------------
This repository contains code and a notebook to analyze how U.S. monetary policy actions and FOMC communications affect USD/VND and USD/CNY (one‑year forward) using event-centered rolling regressions.


Requirements
------------
- Python 3.9 or later
- Recommended environment: virtualenv or conda

Required Python packages:
- numpy
- pandas
- matplotlib
- statsmodels
- scipy
- jupyter (or jupyterlab)

The analysis was developed and tested in a conda environment.
Package versions may slightly affect numerical results but not the main conclusions.



Repository layout
-----------------
- `data/`
  - Contains all datasets used in the analysis.
    Raw data files are provided for transparency and replication.
    Data preprocessing and merging are handled by the `load_data()` function.

- `src/`
  - `event_regression.ipynb` — primary analysis notebook that runs the full pipeline (data loading → event rolling regressions → summary tables → plots).
  - `utils/` — helper modules used by the notebook:
    - `__init__.py`
    - `data_merger.py` — functions to load, clean and merge input data into a single DataFrame.
    - `event_rolling.py` — `EventRolling` and `EventRollingConfig`, functions to run asymmetric rolling event regressions and summarize results.


How to run the analysis (notebook)
----------------------------------
1. Open `src/event_regression.ipynb` in VS Code or JupyterLab.
2. Select the Python kernel that matches the virtual environment you created.
3. Run cells top-to-bottom. The notebook workflow is roughly:
   - import libraries and utils
   - `df = load_data()` (from `utils.data_merger`)
   - exploratory data checks and plots
   - configure `EventRollingConfig` and instantiate `EventRolling`
   - call `rolling_event_regression_asymmetric()` and then `summarize_results()`
   - create summary tables and figures