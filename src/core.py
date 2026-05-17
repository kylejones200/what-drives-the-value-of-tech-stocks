"""Core functions for transfer entropy analysis with VECM."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyinform.transferentropy as te
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def download_etf_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download ETF price data from Yahoo Finance."""
    data = yf.download(tickers, start=start, end=end)["Close"].dropna()
    return data


def perform_adf_test(series: pd.Series, name: str) -> float:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series)
    logging.info(f"{name} ADF Statistic: {result[0]:.4f}")
    logging.info(f"{name} p-value: {result[1]:.4f}")
    return result[1]


def perform_johansen_test(data: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    """Perform Johansen cointegration test."""
    johan_test = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    logging.info("Trace statistics:", johan_test.lr1)
    logging.info("Critical values:", johan_test.cvt)
    return johan_test


def fit_vecm_model(
    data: pd.DataFrame,
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    deterministic: str = "co",
):
    """Fit Vector Error Correction Model."""
    vec_model = VECM(
        data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic
    )
    vec_res = vec_model.fit()
    return vec_res


def plot_etf_prices(data: pd.DataFrame, output_path: Path, plot: bool = False):
    """Plot ETF prices"""
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        for col in data.columns:
            ax.plot(data.index, data[col], label=col, linewidth=1.2)

        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Close Price")
        ax.legend(loc="best")

        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def plot_irf(irf, output_path: Path):
    """Plot Impulse Response Functions"""
    irf.plot(orth=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()


def discretize_series(series: pd.Series, bins: int = 3) -> np.ndarray:
    """Discretize series into bins."""
    return pd.qcut(series.rank(method="first"), bins, labels=False).astype(int).values


def compute_transfer_entropy(x: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Compute transfer entropy from x to y."""
    x_series = x.reshape(1, -1)
    y_series = y.reshape(1, -1)
    return te.transfer_entropy(x_series, y_series, k=k)


def compute_rolling_transfer_entropy(
    df: pd.DataFrame, var1: str, var2: str, window: int = 100, bins: int = 3, k: int = 1
) -> tuple[list, list]:
    """Compute rolling transfer entropy."""
    te_1_to_2 = []
    te_2_to_1 = []

    for i in range(window, len(df)):
        sub = df.iloc[i - window : i]
        var1_disc = discretize_series(sub[var1], bins).reshape(1, -1)
        var2_disc = discretize_series(sub[var2], bins).reshape(1, -1)
        try:
            te1 = te.transfer_entropy(var1_disc, var2_disc, k=k)
            te2 = te.transfer_entropy(var2_disc, var1_disc, k=k)
        except Exception:
            te1, te2 = np.nan, np.nan
        te_1_to_2.append(te1)
        te_2_to_1.append(te2)

    return te_1_to_2, te_2_to_1
