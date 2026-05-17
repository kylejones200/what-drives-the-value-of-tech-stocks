"""Generated from Jupyter notebook: Transfer Entropy with tech etfs

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyinform.transferentropy as te
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from tqdm import tqdm


def adf_test(series, name):
    result = adfuller(series)
    print(f"{name} ADF Statistic: {result[0]:.4f}")
    print(f"{name} p-value: {result[1]:.4f}")
    return result[1]


def adjust_spines(ax):
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def discretize(series, bins=3):
    return pd.qcut(series.rank(method="first"), bins, labels=False)


def main() -> None:
    tickers = ["XLK", "SMH"]

    data = yf.download(tickers, start="2015-01-01", end="2024-12-31")["Close"].dropna()

    data.columns = ["Tech_ETF", "Semiconductor_ETF"]

    data.to_csv("xlk_smh_prices.csv")

    df = pd.read_csv("xlk_smh_prices.csv", parse_dates=["Date"], index_col="Date")

    df.plot(
        title="Tech ETF (XLK) vs Semiconductor ETF (SMH)", ylabel="Adjusted Close Price"
    )

    plt.tight_layout()

    plt.savefig("xlk_smh_prices.png")

    plt.show()

    adf_test(df["Tech_ETF"], "XLK")

    adf_test(df["Semiconductor_ETF"], "SMH")

    johan_test = coint_johansen(df, det_order=0, k_ar_diff=1)

    print(johan_test.lr1)

    print(johan_test.cvt)

    vec_model = VECM(df, k_ar_diff=1, coint_rank=1, deterministic="co")

    vec_res = vec_model.fit()

    print(vec_res.summary())

    irf = vec_res.irf(10)

    irf.plot(orth=False)

    plt.tight_layout()

    plt.savefig("vecm_irf.png")

    plt.show()

    df = pd.read_csv("xlk_smh_prices.csv", parse_dates=["Date"], index_col="Date")

    df_diff = df.diff().dropna()

    df_diff.plot(title="Differenced Prices: XLK vs SMH", ylabel="First Differences")

    plt.tight_layout()

    plt.show()

    xlk_disc = discretize(df_diff["Tech_ETF"]).astype(int).values

    smh_disc = discretize(df_diff["Semiconductor_ETF"]).astype(int).values

    xlk_series = xlk_disc.reshape(1, -1)

    smh_series = smh_disc.reshape(1, -1)

    te_smh_to_xlk = te.transfer_entropy(smh_series, xlk_series, k=1)

    te_xlk_to_smh = te.transfer_entropy(xlk_series, smh_series, k=1)

    print(f"Transfer Entropy SMH → XLK: {te_smh_to_xlk:.4f}")

    print(f"Transfer Entropy XLK → SMH: {te_xlk_to_smh:.4f}")

    df = pd.read_csv("xlk_smh_prices.csv", parse_dates=["Date"], index_col="Date")

    df_diff = df.diff().dropna()

    window = 100

    bins = 3

    te_smh_to_xlk = []

    te_xlk_to_smh = []

    dates = []

    for i in tqdm(range(window, len(df_diff))):
        sub = df_diff.iloc[i - window : i]
        xlk_disc = discretize(sub["Tech_ETF"], bins).astype(int).values.reshape(1, -1)
        smh_disc = (
            discretize(sub["Semiconductor_ETF"], bins).astype(int).values.reshape(1, -1)
        )
        try:
            te1 = te.transfer_entropy(smh_disc, xlk_disc, k=1)
            te2 = te.transfer_entropy(xlk_disc, smh_disc, k=1)
        except Exception:
            te1, te2 = (np.nan, np.nan)
        te_smh_to_xlk.append(te1)
        te_xlk_to_smh.append(te2)
        dates.append(sub.index[-1])

    te_df = pd.DataFrame(
        {"SMH → XLK": te_smh_to_xlk, "XLK → SMH": te_xlk_to_smh},
        index=pd.to_datetime(dates),
    )

    te_df.plot(title="Rolling Transfer Entropy (100-day Window)")

    plt.ylabel("Transfer Entropy")

    plt.tight_layout()

    plt.savefig("rolling_transfer_entropy.png")

    plt.show()

    features = pd.DataFrame(index=te_df.index)

    features["TE_SMH_to_XLK"] = te_df["SMH → XLK"]

    features["TE_XLK_to_SMH"] = te_df["XLK → SMH"]

    df_ret = df.diff().dropna().pct_change().dropna()

    features["SMH_Return"] = df_ret.loc[features.index, "Semiconductor_ETF"]

    target = df_ret.loc[features.index, "Tech_ETF"]

    data = pd.concat([features, target.rename("XLK_Return")], axis=1)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    data.dropna(inplace=True)

    train = data.iloc[:-200]

    test = data.iloc[-200:]

    model = LinearRegression()

    model.fit(train.drop("XLK_Return", axis=1), train["XLK_Return"])

    y_pred = model.predict(test.drop("XLK_Return", axis=1))

    r2 = r2_score(test["XLK_Return"], y_pred)

    rmse = mean_squared_error(test["XLK_Return"], y_pred)

    print(f"Test R²: {r2:.4f}, RMSE: {rmse:.6f}")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(data.drop("XLK_Return", axis=1))

    y_scaled = scaler.fit_transform(data[["XLK_Return"]])

    X_train, X_test = (X_scaled[:-200], X_scaled[-200:])

    y_train, y_test = (y_scaled[:-200], y_scaled[-200:])

    model = LinearRegression()

    model.fit(X_train, y_train.ravel())

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    rmse = mean_squared_error(y_test, y_pred)

    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}")

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

    rf.fit(X_train, y_train.ravel())

    y_pred_rf = rf.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)

    rmse_rf = mean_squared_error(y_test, y_pred_rf)

    print(f"Random Forest R²: {r2_rf:.4f}, RMSE: {rmse_rf:.4f}")

    df = pd.read_csv("xlk_smh_prices.csv", parse_dates=["Date"], index_col="Date")

    df_diff = df.diff().dropna()

    xlk_disc = discretize(df_diff["Tech_ETF"]).astype(int).values.reshape(1, -1)

    smh_disc = (
        discretize(df_diff["Semiconductor_ETF"]).astype(int).values.reshape(1, -1)
    )

    te_smh_to_xlk = te.transfer_entropy(smh_disc, xlk_disc, k=1)

    te_xlk_to_smh = te.transfer_entropy(xlk_disc, smh_disc, k=1)

    print(f"SMH → XLK: {te_smh_to_xlk:.4f}")

    print(f"XLK → SMH: {te_xlk_to_smh:.4f}")

    window = 100

    bins = 3

    te_smh_to_xlk = []

    te_xlk_to_smh = []

    dates = []

    for i in tqdm(range(window, len(df_diff))):
        sub = df_diff.iloc[i - window : i]
        xlk = discretize(sub["Tech_ETF"], bins).astype(int).values.reshape(1, -1)
        smh = (
            discretize(sub["Semiconductor_ETF"], bins).astype(int).values.reshape(1, -1)
        )
        try:
            te1 = te.transfer_entropy(smh, xlk, k=1)
            te2 = te.transfer_entropy(xlk, smh, k=1)
        except Exception:
            te1, te2 = (np.nan, np.nan)
        te_smh_to_xlk.append(te1)
        te_xlk_to_smh.append(te2)
        dates.append(sub.index[-1])

    te_df = pd.DataFrame(
        {"SMH → XLK": te_smh_to_xlk, "XLK → SMH": te_xlk_to_smh},
        index=pd.to_datetime(dates),
    )

    te_df.plot(title="Rolling Transfer Entropy (100-Day Window)", figsize=(12, 6))

    plt.ylabel("Transfer Entropy")

    plt.tight_layout()

    plt.savefig("rolling_transfer_entropy.png")

    plt.show()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(te_df.index, te_df["SMH → XLK"], label="SMH → XLK", color="black")

    ax.plot(te_df.index, te_df["XLK → SMH"], label="XLK → SMH", color="orange")

    adjust_spines(ax)

    ax.set_title("Rolling Transfer Entropy (100-Day Window)", fontsize=14)

    ax.set_ylabel("Transfer Entropy", fontsize=12)

    ax.set_xlabel("Date", fontsize=12)

    ax.legend(frameon=False)

    plt.tight_layout()

    plt.savefig("rolling_transfer_entropy_minimalist.png")

    plt.show()

    file_path = "xlk_smh_prices.csv"

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    df_diff = df.diff().dropna()

    window = 100

    bins = 3

    te_smh_to_xlk = []

    te_xlk_to_smh = []

    dates = []

    for i in tqdm(range(window, len(df_diff))):
        sub = df_diff.iloc[i - window : i]
        xlk = discretize(sub["Tech_ETF"], bins).astype(int).values.reshape(1, -1)
        smh = (
            discretize(sub["Semiconductor_ETF"], bins).astype(int).values.reshape(1, -1)
        )
        try:
            te1 = te.transfer_entropy(smh, xlk, k=1)
            te2 = te.transfer_entropy(xlk, smh, k=1)
        except Exception:
            te1, te2 = (np.nan, np.nan)
        te_smh_to_xlk.append(te1)
        te_xlk_to_smh.append(te2)
        dates.append(sub.index[-1])

    te_df = pd.DataFrame(
        {"SMH → XLK": te_smh_to_xlk, "XLK → SMH": te_xlk_to_smh},
        index=pd.to_datetime(dates),
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(te_df.index, te_df["SMH → XLK"], label="SMH → XLK", color="black")

    ax.plot(te_df.index, te_df["XLK → SMH"], label="XLK → SMH", color="gray")

    adjust_spines(ax)

    ax.set_title("Rolling Transfer Entropy (100-Day Window)", fontsize=14)

    ax.set_ylabel("Transfer Entropy", fontsize=12)

    ax.set_xlabel("Date", fontsize=12)

    ax.legend(frameon=False)

    plt.tight_layout()

    plt.savefig("rolling_transfer_entropy_minimalist.png")

    plt.show()

    te_df_smooth = te_df.resample("ME").mean()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        te_df_smooth.index, te_df_smooth["SMH → XLK"], label="SMH → XLK", color="black"
    )

    ax.plot(
        te_df_smooth.index, te_df_smooth["XLK → SMH"], label="XLK → SMH", color="red"
    )

    adjust_spines(ax)

    ax.set_title(
        "Rolling Transfer Entropy (100-Day TE, Monthly Moving Average MA)", fontsize=14
    )

    ax.legend(frameon=False)

    plt.tight_layout()

    plt.savefig("rolling_transfer_entropy_smoothed.png")

    plt.show()


if __name__ == "__main__":
    main()
