---
author: "Kyle Jones"
date_published: "April 30, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/what-drives-the-value-of-tech-stocks-d86ee4f7b370"
---

# What Drives the Value of Tech Stocks? Some ETFs just move together. XLK tracks big tech. SMH tracks
chipmakers. You can eyeball the chart and see they're correlated. But...

### What Drives the Value of Tech Stocks? Nonlinear Causality Between High Tech and Semiconductor ETFs(time series with python)
Some ETFs just move together. XLK tracks big tech. SMH tracks chipmakers. You can eyeball the chart and see they're correlated. But correlation isn't the question. The question is: which one leads?

We used Transfer Entropy, a nonlinear, model-free technique from information theory, to test directional influence. No assumptions. No models. Just signal flow: does knowing SMH's past improve your knowledge of XLK's future?

It turns out the answer is yes --- consistently, measurably, and especially when it matters most.

Most causality tests assume linear relationships. Transfer Entropy doesn't. It doesn't care if the relationship is linear, quadratic, volatile, or buried under noise. It measures whether there's **additional information** in one time series that helps explain the next state of another. That's it.

Mathematically, it compares two conditional entropies:

- How predictable is XLK from its own past?
- How much more predictable is XLK if you also include SMH?

If SMH adds information, Transfer Entropy shows it. And it works in one direction at a time, so you can test asymmetry.

We pulled 10 years of daily adjusted close data for XLK and SMH from Yahoo Finance (2015 to 2024). Then we:

1.  [Took first differences to make the series stationary.]
2.  [Discretized the returns into 3 bins: down, flat, up.]
3.  [Measured Transfer Entropy from SMH to XLK, and vice versa.]

Across the full 10-year period, Transfer Entropy:

- **SMH → XLK**: 0.0205
- **XLK → SMH**: 0.0190

The difference is small --- but persistent. Chipmakers transmit more information into the tech index than the other way around. That tracks with intuition. SMH reacts quickly to supply chain issues, innovation shocks, and fabrication risk. XLK reflects those shifts later.

### Rolling Transfer Entropy: Seeing the Structure
To move beyond a single number, we measured Transfer Entropy in a **rolling 100-day window**, then applied a **10-day moving average** to smooth the results. The result is a causal time series. You can now see when one ETF starts to lead or lag the other, and how the strength of influence changes with market conditions.

Here's the plot:


SMH consistently leads XLK across nearly the entire decade. The strength of that lead widens in volatile periods --- early 2020, mid-2022, and again in late 2023. In less volatile markets, the TE values flatten, but the direction doesn't flip. SMH never becomes passive.

This shows more than correlation. This is a time-structured information flow. You now know when the semiconductor sector starts moving first --- and when that signal fades.

### Why This Matters
Most financial models fail because they confuse noise with signal. What we showed instead is that signal exists, and it moves directionally. You don't need to build a model to benefit from that.

Don't assume symmetry in the relationship. XLK doesn't give you the same clarity about SMH. Information flows one way more than the other.

### If You Want to Run This Yourself
Here's the full Python code to get the plot using `pyinform`:

```python
import pandas as pd
import numpy as np
import pyinform.transferentropy as te
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf


# Define tickers and time range
tickers = ['XLK', 'SMH']
data = yf.download(tickers, start="2015-01-01", end="2024-12-31")['Close'].dropna()
data.columns = ['Tech_ETF', 'Semiconductor_ETF']

# Save for analysis
data.to_csv("xlk_smh_prices.csv")


# Load and differenced data
df = pd.read_csv("xlk_smh_prices.csv", parse_dates=["Date"], index_col="Date")
df_diff = df.diff().dropna()
# Discretize to 3 bins (down, flat, up)
def discretize(series, bins=3):
    return pd.qcut(series.rank(method="first"), bins, labels=False)
# Rolling TE computation
window = 100
te_smh_to_xlk, te_xlk_to_smh, dates = [], [], []
for i in tqdm(range(window, len(df_diff))):
    sub = df_diff.iloc[i - window:i]
    xlk = discretize(sub["Tech_ETF"]).astype(int).values.reshape(1, -1)
    smh = discretize(sub["Semiconductor_ETF"]).astype(int).values.reshape(1, -1)
    try:
        te1 = te.transfer_entropy(smh, xlk, k=1)
        te2 = te.transfer_entropy(xlk, smh, k=1)
    except:
        te1, te2 = np.nan, np.nan
    te_smh_to_xlk.append(te1)
    te_xlk_to_smh.append(te2)
    dates.append(sub.index[-1])
# Store and smooth
te_df = pd.DataFrame({
    "SMH → XLK": te_smh_to_xlk,
    "XLK → SMH": te_xlk_to_smh
}, index=pd.to_datetime(dates))
te_df = te_df.rolling(window=10, min_periods=1).mean()
```

Minimalist Plot

```python
plt.rcParams.update({
    'axes.grid': False,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

def adjust_spines(ax):
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(te_df.index, te_df["SMH → XLK"], label="SMH → XLK", color="black")
ax.plot(te_df.index, te_df["XLK → SMH"], label="XLK → SMH", color="gray")
adjust_spines(ax)
ax.set_title("Smoothed Rolling Transfer Entropy (100-Day TE, 10-Day MA)", fontsize=14)
ax.set_ylabel("Transfer Entropy", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("rolling_transfer_entropy_smoothed.png")
plt.show()
```
