import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EventRollingConfig:
    window_size: int = 21
    days_before: int = 21
    days_after: int = 21
    cov_type: Literal[
        "nonrobust", "fixed scale", "HC0", "HC1", "HC2", "HC3",
        "HAC", "hac-panel", "hac-groupsum", "cluster"
    ] = "HC3"
    cov_kwargs: dict = field(default_factory=dict)
    significance_level: float = 0.05
    plot_confidence_int: bool = True
    shade_event_window: bool = True
    lags: List[int] = field(default_factory=list)
    event_col: str = "fomc_event"
    event_indicator_col: str = "fomc_event"


class EventRolling:
    def __init__(
        self,
        df: pd.DataFrame,
        x_lagged_vars: List[str],
        x_non_lagged_vars: List[str],
        y_vars: List[str],
        config: EventRollingConfig = None,
    ):
        self.config = config or EventRollingConfig()
        self.df = df.copy().ffill()

        self.y_vars = y_vars
        self.x_lagged_vars = x_lagged_vars
        self.x_non_lagged_vars = x_non_lagged_vars

        lagged_vars = []

        for var in self.x_lagged_vars:
            for lag in self.config.lags:
                if lag == 0:
                    col_name = f"{var}_t"
                    self.df[col_name] = self.df[var]
                elif lag < 0:
                    col_name = f"{var}_lag{abs(lag)}"
                    self.df[col_name] = self.df[var].shift(abs(lag))
                else:
                    col_name = f"{var}_lead{lag}"
                    self.df[col_name] = self.df[var].shift(-lag)
                lagged_vars.append(col_name)

        self.indep_vars = self.x_non_lagged_vars + lagged_vars

        if self.df[self.config.event_col].nunique() <= 2:
            self._event_idx = self.df[self.df[self.config.event_col] == 1].index
        else:
            self._event_idx = self.df[self.df[self.config.event_col].notna()].index

        self._X_full = self.df[self.indep_vars]

    def _run_rolling_regression(self, df: pd.DataFrame, before: int, after: int) -> pd.DataFrame:
        events = self._event_idx
        data = df.copy().ffill()
        results = []

        for idx in events:
            start = max(0, idx - before)
            end = min(len(data), idx + after + 1)

            subset_cols = self.indep_vars + self.y_vars
            window = data.iloc[start:end].dropna(subset=subset_cols)

            X = window[self.indep_vars]
            y = window[self.y_vars[0]]

            if X.empty or y.empty or len(X) != len(y):
                continue
            if len(X) <= X.shape[1]:
                continue  

            try:
                model = sm.OLS(y, X).fit(
                    cov_type=self.config.cov_type,
                    cov_kwds=self.config.cov_kwargs
                )
            except Exception:
                continue  

            entry: Dict[str, Any] = {
                "event_index": idx,
                "event_date": (
                    data.loc[idx, "date"]
                    if "date" in data.columns
                    else data.index[idx]
                ),
                "r_squared": model.rsquared
            }

            for var in X.columns:
                entry[f"beta_{var}"] = model.params.get(var, np.nan)
                p = model.pvalues.get(var, np.nan)
                entry[f"pval_{var}"] = p
                entry[f"significant_{var}"] = p < self.config.significance_level

            results.append(entry)

        return pd.DataFrame(results)

    def rolling_event_regression(self) -> pd.DataFrame:
        ws = self.config.window_size
        return self._run_rolling_regression(self.df, ws, ws)

    def rolling_event_regression_asymmetric(self) -> pd.DataFrame:
        return self._run_rolling_regression(
            self.df,
            self.config.days_before,
            self.config.days_after
        )

    def summarize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        summary = []
        alpha = self.config.significance_level
        total_events = len(results_df)

        beta_cols = [c for c in results_df.columns if c.startswith("beta_")]

        for beta_col in beta_cols:
            var = beta_col.replace("beta_", "")
            pval_col = f"pval_{var}"

            df_var = results_df[[beta_col, pval_col, "event_date"]].dropna()
            sig_df = df_var[df_var[pval_col] < alpha]

            significant_count = len(sig_df)
            ratio = significant_count / total_events if total_events > 0 else np.nan

            if sig_df.empty:
                max_date = np.nan
                max_beta = np.nan
            else:
                max_idx = sig_df[beta_col].abs().idxmax()
                max_date = sig_df.loc[max_idx, "event_date"]
                max_beta = sig_df.loc[max_idx, beta_col]

            summary.append({
                "variable": var,
                "significant_count": significant_count,
                "total_events": total_events,
                "significant_ratio": round(ratio, 3),
                "mean_beta": sig_df[beta_col].mean() if not sig_df.empty else np.nan,
                "mean_pval": sig_df[pval_col].mean() if not sig_df.empty else np.nan,
                "max_abs_beta": abs(max_beta) if not pd.isna(max_beta) else np.nan,
                "max_beta_date": max_date
            })

        return pd.DataFrame(summary).sort_values(by="significant_count", ascending=False)

    def plot_event_quality(self, results_df: pd.DataFrame, r2_threshold: float = 0.05) -> None:
        if results_df.empty:
            print("No data available for plotting.")
            return

        sig_cols = [c for c in results_df.columns if c.startswith("significant_")]
        results_df["significant_beta_count"] = results_df[sig_cols].sum(axis=1)

        results_df["is_effective"] = (
            (results_df["r_squared"] >= r2_threshold) |
            (results_df["significant_beta_count"] > 0)
        )

        dates = pd.to_datetime(results_df["event_date"])
        r2 = results_df["r_squared"]
        effective = results_df["is_effective"]

        plt.figure(figsize=(14, 5))
        plt.plot(dates, r2, marker="o", label="R²", color="blue")
        plt.axhline(
            r2_threshold,
            color="red",
            linestyle="--",
            label=f"R² threshold = {r2_threshold}"
        )

        plt.scatter(
            dates[effective],
            r2[effective],
            color="green",
            label="Effective Events",
            s=100,
            zorder=3
        )

        plt.title("Event effectiveness based on R² and statistical significance", fontsize=14)
        plt.xlabel("Event date")
        plt.ylabel("R-squared")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
