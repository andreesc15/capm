import yfinance as yf
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sb

# Markowitz
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.expected_returns import mean_historical_return

# CAPM
from scipy import stats

#===========================================================#
# 0. DATA PULL
#===========================================================#
price_df = yf.download("GGRM.JK CTRA.JK UNTR.JK BRPT.JK TLKM.JK INCO.JK BBNI.JK AMRT.JK MIKA.JK", start="2022-07-31", end="2024-10-30", interval='1mo')['Close']
jkse_df  = yf.download("^JKSE", start="2022-07-31", end="2024-10-30", interval='1mo')['Close']
#lq45_df  = yf.download("^JKLQ45", start="2022-07-31", end="2024-10-30", interval='1mo')['Close']
ticker_list = price_df.columns.values
# print(ticker_list)

log_return_df = np.log(price_df / price_df.shift(1))
log_return_df = log_return_df.dropna()

jkse_return_df = np.log(jkse_df / jkse_df.shift(1))
jkse_return_df = jkse_return_df.dropna()
# print(log_return_df)

#===========================================================#
# 1. DESCRIPTIVE STATISTICS
#===========================================================#
# Wrapped in function, making it easier to call or not to call.
def call_descriptive():
    price_summary = price_df.describe().round(4)
    lr_summary = log_return_df.describe().round(4)
    print(price_summary)
    print(lr_summary)

    #------- PRICE MOVEMENT DATA
    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axs = axs.flatten()

    for i, ticker in enumerate(ticker_list):
        ax = axs[i]
        ax2 = ax.twinx()
        
        # Plot stock price (left y-axis)
        ax.plot(price_df.index, price_df[ticker], color='red', label=f'{ticker} Harga Saham')
        ax.set_ylabel('Harga Saham', color='red')
        ax.tick_params(axis='y', labelcolor='red', labelsize=8)
        
        # Plot log returns (right y-axis)
        ax2.plot(log_return_df.index, log_return_df[ticker], color='blue', linestyle='--', label=f'{ticker} Log Return')
        ax2.plot(jkse_return_df.index, jkse_return_df, color='gray', linestyle=':', label='IHSG Log Return')
        ax2.set_ylabel('Log Return', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)
        
        # Titles
        ax.set_title(f'{ticker}', fontsize=10)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.set_xlim([price_df.index.min(), price_df.index.max()])
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Legends
        ax.legend(loc='upper left', fontsize=7, frameon=True)
        ax2.legend(loc='upper right', fontsize=7, frameon=True)
        
        # Grid
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Hide unused subplots (if any)
    for j in range(len(ticker_list), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

    corr = log_return_df.corr()
    print(corr)
    plt.figure(figsize=(10,8))
    sb.heatmap(corr, cmap="Blues", annot=True, fmt=".1f", annot_kws={"size": 9})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Korelasi antar Saham", fontsize=14)
    plt.tight_layout()
    plt.show()

#===========================================================#
# 2. MARKOWITZ EfficientFrontier PORTOFOLIO OPTIMIZATION
#===========================================================#
mu = log_return_df.mean()
S = log_return_df.cov()

rf = 0.05443326795 / 12 # 5.443326795% dari average INDONIA sesuai dengan periode, dijadikan MONTLY RATE karena datanya bulanan


delta       = 1.0                            # risk-aversion (for max_quadratic_utility)
target_vol  = log_return_df.std().mean()     # simple target σ
target_ret  = mu.mean()                      # simple target µ

optimisers = {
    "min_volatility"        : lambda ef: ef.min_volatility(),
    "max_sharpe"            : lambda ef: ef.max_sharpe(risk_free_rate=rf),
    "max_quadratic_utility" : lambda ef: ef.max_quadratic_utility(risk_aversion=delta),
    "efficient_risk"        : lambda ef: ef.efficient_risk(target_volatility=target_vol),
    "efficient_return"      : lambda ef: ef.efficient_return(target_return=target_ret),
}

bound_sets = {
    "wo_shorts": (0, 1),
    "with_shorts": (-10, 10),
}

opt_results  = {}   # FLAT  —>  { "max_sharpe_wo_shorts": weights, ... }
perf_results = {}   # optional: same keys but (ret, vol, sharpe) tuples

for b_label, bounds in bound_sets.items():
    for opt_name, opt_fun in optimisers.items():
        key = f"{opt_name}_{b_label}"
        ef  = EfficientFrontier(mu, S, weight_bounds=bounds)
        opt_results[key]  = opt_fun(ef)                          # store WEIGHTS dict
        perf_results[key] = ef.portfolio_performance(            # (mu, sigma, Sharpe)
                                verbose=False, risk_free_rate=rf)

        # Console feedback (optional)
        mu_, sigma_, sharpe_ = perf_results[key]
        print(f"{key:<30}  μ={mu_:6.3%}  σ={sigma_:6.3%}  SR={sharpe_:5.2f}")

# Keep the two explicit variables your CAPM loop expects
ef_no_shorts_weights = opt_results["max_sharpe_wo_shorts"]
ef_w_shorts_weights  = opt_results["max_sharpe_with_shorts"]

#------ linear regression helper function
def reg_model(x, y):
    reg = stats.linregress(x, y)
    return {
        "slope"             : reg.slope,
        "slope_stderr"      : reg.stderr,       # stderr of the slope
        "slope_pvalue"      : reg.pvalue,       # two‐sided p‐value for testing slope:0

        "intercept"         : reg.intercept,
        "intercept_stderr"  : reg.intercept_stderr,
        "intercept_pvalue"  : 2 * stats.t.sf(np.abs(reg.intercept / reg.intercept_stderr), df = len(x) - 2),
        "rvalue"            : reg.rvalue,
        "r_squared"         : reg.rvalue ** 2,
    }
# ---------- pretty printer for weight dictionaries ----------
def weights_to_df(results_dict, tickers):
    """
    Convert {model: {ticker: weight, …}, …}  →  DataFrame
    with models as rows and tickers as columns, rounded to 4 dp.
    """
    # Build DataFrame from the nested dict, re-index columns to desired order
    df = pd.DataFrame(results_dict).T.reindex(columns=tickers)
    return df.round(4)

print(weights_to_df(opt_results, ticker_list))
#===========================================================#
# 3. CAPM
#===========================================================#
# --------- INDIVIDIUAL & MULTIPLE ASSET  CAPM ANALYTICS -------
capm_regression_asset_list = []
for model, weight in opt_results.items():
    x = jkse_return_df['^JKSE'].subtract(rf)

    w_array = np.array([weight[t] for t in ticker_list]).reshape((9, 1))
    y = (log_return_df.dot(w_array)) - (rf)
    y = y.values.flatten()

    reg_dict = reg_model(x, y)
    reg_dict['model'] = model

    #------- compute TR & ER
    reg_dict['ER'] = float(log_return_df.dot(w_array).mean() - rf)
    reg_dict['TR'] = float(reg_dict['ER'] / reg_dict['slope'])

    capm_regression_asset_list.append(reg_dict)

for ticker in ticker_list:
    x = jkse_return_df['^JKSE'].subtract(rf)
    y = log_return_df[ticker].subtract(rf)

    reg_dict = reg_model(x, y)
    reg_dict['model'] = ticker

    #------- compute TR & ER
    reg_dict['ER'] = log_return_df[ticker].mean() - rf
    reg_dict['TR'] = reg_dict['ER'] / reg_dict['slope']

    capm_regression_asset_list.append(reg_dict)

results_df = pd.DataFrame(capm_regression_asset_list)
print(results_df)
