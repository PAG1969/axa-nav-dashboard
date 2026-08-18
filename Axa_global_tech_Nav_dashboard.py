from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
import yfinance as yf

st.title("AXA Global Technology NAV Predictor")


def get_top_10_holdings(target_date=None):
  if target_date is not None:
    target_date = pd.to_datetime(target_date)

  cutoff_date = pd.to_datetime('2026-08-01')

  if target_date is not None and target_date < cutoff_date:
    return pd.DataFrame({
        'Name': [
            'SK Hynix',
            'NVIDIA',
            'TSMC',
            'Alphabet Class C',
            'Micron',
            'Lam Research',
            'Broadcom',
            'Microsoft',
            'Apple',
            'AMD',
        ],
        'Ticker': [
            '000660.KS',
            'NVDA',
            '2330.TW',
            'GOOG',
            'MU',
            'LRCX',
            'AVGO',
            'MSFT',
            'AAPL',
            'AMD',
        ],
        'Weight': [
            0.0750,
            0.0700,
            0.0600,
            0.0550,
            0.0480,
            0.0400,
            0.0390,
            0.0350,
            0.0300,
            0.0280,
        ],
        'Currency': [
            'KRW',
            'USD',
            'TWD',
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
        ],
    })
  else:
    return pd.DataFrame({
        'Name': [
            'SK Hynix',
            'NVIDIA',
            'TSMC',
            'Alphabet Class C',
            'Micron',
            'Lam Research',
            'Broadcom',
            'Astera Labs',
            'Jfrog',
            'AMD',
        ],
        'Ticker': [
            '000660.KS',
            'NVDA',
            '2330.TW',
            'GOOG',
            'MU',
            'LRCX',
            'AVGO',
            'ALAB',
            'FROG',
            'AMD',
        ],
        'Weight': [
            0.0696,
            0.0658,
            0.0587,
            0.0537,
            0.0454,
            0.0396,
            0.0386,
            0.0344,
            0.0304,
            0.0280,
        ],
        'Currency': [
            'KRW',
            'USD',
            'TWD',
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
            'USD',
        ],
    })


start_date = '2026-07-01'
end_date = '2026-08-18'

actual_dates = [
    '2026-07-13',
    '2026-07-14',
    '2026-07-15',
    '2026-07-16',
    '2026-07-17',
    '2026-07-20',
    '2026-07-21',
    '2026-07-22',
    '2026-07-23',
    '2026-07-24',
    '2026-07-27',
    '2026-07-28',
    '2026-07-29',
    '2026-07-30',
    '2026-07-31',
    '2026-08-03',
    '2026-08-04',
    '2026-08-05',
    '2026-08-06',
    '2026-08-07',
    '2026-08-10',
    '2026-08-11',
    '2026-08-12',
    '2026-08-13',
    '2026-08-14',
    '2026-08-17',
]

actual_nav_path = [
    4.26,
    4.17,
    4.24,
    4.14,
    3.96,
    3.96,
    4.06,
    4.11,
    4.09,
    4.02,
    3.99,
    3.89,
    3.77,
    3.68,
    3.93,
    3.90,
    3.99,
    4.17,
    4.10,
    4.12,
    4.20,
    4.16,
    4.16,
    4.24,
    4.2860,
    4.2690,
]

masked_dates = {'2026-07-30'}

current_holdings = get_top_10_holdings(pd.to_datetime('2026-08-17'))
past_holdings = get_top_10_holdings(pd.to_datetime('2026-07-15'))

tickers_list = list(
    set(current_holdings['Ticker'].tolist() + past_holdings['Ticker'].tolist())
)
tickers_list.extend(
    ['SOXX', 'QQQ', 'IGV', 'META', 'AMZN', 'TSLA', 'KRW=X', 'TWD=X']
)

data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)[
    'Close'
]
returns_df = data.pct_change(fill_method=None) 
fx_returns = data[['KRW=X', 'TWD=X']].pct_change(fill_method=None)

if '000660.KS' in returns_df.columns and 'KRW=X' in fx_returns.columns:
  krw_ret = fx_returns['KRW=X'].shift(1).fillna(0)
  returns_df['000660.KS'] = ((1 + returns_df['000660.KS']) / (1 + krw_ret)) - 1

if '2330.TW' in returns_df.columns and 'TWD=X' in fx_returns.columns:
  twd_ret = fx_returns['TWD=X'].shift(1).fillna(0)
  returns_df['2330.TW'] = ((1 + returns_df['2330.TW']) / (1 + twd_ret)) - 1

for t in ['000660.KS', '2330.TW']:
  if t in returns_df.columns:
    returns_df[t] = returns_df[t].shift(1)

returns_df = returns_df.fillna(0.0)

available_dates = [
    d for d in actual_dates if d in returns_df.index.strftime('%Y-%m-%d').tolist()
]
actual_nav_dict = dict(zip(actual_dates, actual_nav_path)) 
matched_dates = [d for d in available_dates if d in actual_nav_dict]

soxx_ret = returns_df['SOXX'].fillna(0.0) 
igv_ret = returns_df['IGV'].fillna(0.0) 
core_residual_ret = (
    returns_df['META'].fillna(0.0)
    + returns_df['AMZN'].fillna(0.0)
    + returns_df['TSLA'].fillna(0.0)
) / 3.0

daily_fee_drag = 0.0068 / 365.0


def objective(weights):
  w_soxx, w_igv, w_core = weights
  errors = []
  for i in range(1, len(matched_dates)):
    d_str = matched_dates[i]
    idx_date = pd.to_datetime(d_str)
    if idx_date not in returns_df.index:
      continue
    day_top_10 = get_top_10_holdings(idx_date)
    t1_w = day_top_10['Weight'].sum()
    t2_w = 1.0 - t1_w

    t1_ret_val = sum(
        (row['Weight'] / t1_w) * returns_df[row['Ticker']].loc[idx_date]
        for _, row in day_top_10.iterrows()
    )
    t2_ret_val = (
        (w_soxx * soxx_ret.loc[idx_date])
        + (w_igv * igv_ret.loc[idx_date])
        + (w_core * core_residual_ret.loc[idx_date])
    )
    gross_return = (t1_w * t1_ret_val) + (t2_w * t2_ret_val) - daily_fee_drag
    next_nav = (
        actual_nav_dict[matched_dates[0]]
        if i == 1
        else next_nav * (1.0 + gross_return)
    )

    if idx_date >= pd.to_datetime('2026-08-01') and d_str not in masked_dates:
      actual_val = actual_nav_dict[d_str]
      te_pct = abs((next_nav - actual_val) / actual_val) * 100
      errors.append(te_pct)
  return np.mean(errors) if errors else 0.0


result = minimize(
    objective,
    [0.4, 0.3, 0.3],
    method='SLSQP',
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
)
opt_soxx, opt_igv, opt_core = result.x

backtest_data = []
current_model_nav = actual_nav_dict[matched_dates[0]]

for i, d_str in enumerate(matched_dates):
  actual_val = actual_nav_dict[d_str]
  if i == 0:
    te_pct = 0.0
    modeled_val = actual_val
  else:
    idx_date = pd.to_datetime(d_str)
    day_top_10 = get_top_10_holdings(idx_date)
    t1_w = day_top_10['Weight'].sum()
    t2_w = 1.0 - t1_w

    t1_ret_val = sum(
        (row['Weight'] / t1_w) * returns_df[row['Ticker']].loc[idx_date]
        for _, row in day_top_10.iterrows()
    )
    t2_ret_val = (
        (opt_soxx * soxx_ret.loc[idx_date])
        + (opt_igv * igv_ret.loc[idx_date])
        + (opt_core * core_residual_ret.loc[idx_date])
    )
    gross_return = (t1_w * t1_ret_val) + (t2_w * t2_ret_val) - daily_fee_drag
    current_model_nav = current_model_nav * (1.0 + gross_return)
    modeled_val = current_model_nav
    te_pct = abs((modeled_val - actual_val) / actual_val) * 100

  backtest_data.append({
      'Date': d_str,
      'Actual NAV': actual_val,
      'Modeled NAV': round(modeled_val, 4),
      'Error (%)': round(te_pct, 2),
  })
backtest_df = pd.DataFrame(backtest_data) 
st.subheader("Backtest Performance Table") 
st.dataframe(backtest_df, use_container_width=True)

st.subheader("Modeled NAV vs. Actual NAV Over Time") 
import plotly.express as px

chart_df = backtest_df.set_index("Date")[["Actual NAV", "Modeled NAV"]] 
fig = px.line(chart_df, labels={"value": "NAV (£)", "Date": "Date"}) 
fig.update_layout(yaxis_tickprefix="£", yaxis_title="NAV (£)") 
st.plotly_chart(fig, use_container_width=True)


live_top_10 = get_top_10_holdings()
tier1_weight = live_top_10['Weight'].sum() 
tier2_weight = 1.0 - tier1_weight

live_tickers = live_top_10['Ticker'].tolist() + [
    'SOXX',
    'IGV',
    'META',
    'AMZN',
    'TSLA',
    'KRW=X',
    'TWD=X',
    'NQ=F',
    'ES=F',
]
live_data = yf.download(live_tickers, period='5d', interval='1d', progress=False)[
    'Close'
]
live_data = live_data.ffill().dropna(how='all')

gia_units = 8970.39
isa_units = 4581.90
total_units = gia_units + isa_units

if not live_data.empty and len(live_data) >= 2:
  prev_close = live_data.iloc[-2]
  latest_premarket = live_data.iloc[-1]
  asset_returns = (latest_premarket - prev_close) / prev_close

  fx_data = live_data[['KRW=X', 'TWD=X']].pct_change().dropna()
  fx_returns = (
      fx_data.iloc[-1]
      if not fx_data.empty
      else pd.Series(0.0, index=['KRW=X', 'TWD=X'])
  )

  if '000660.KS' in asset_returns.index and 'KRW=X' in fx_returns.index:
    asset_returns['000660.KS'] = (
        (1 + asset_returns['000660.KS']) / (1 + fx_returns['KRW=X'])
    ) - 1
  if '2330.TW' in asset_returns.index and 'TWD=X' in fx_returns.index:
    asset_returns['2330.TW'] = (
        (1 + asset_returns['2330.TW']) / (1 + fx_returns['TWD=X'])
    ) - 1

  t1_return = 0.0
  for _, row in live_top_10.iterrows():
    t1_return += (row['Weight'] / tier1_weight) * asset_returns.get(
        row['Ticker'], 0.0
    )

  live_core_residual = (
      asset_returns.get('META', 0.0)
      + asset_returns.get('AMZN', 0.0)
      + asset_returns.get('TSLA', 0.0)
  ) / 3.0

  t2_return = (
      (opt_soxx * asset_returns.get('SOXX', 0.0))
      + (opt_igv * asset_returns.get('IGV', 0.0))
      + (opt_core * live_core_residual)
  )

  nasdaq_futures_sentiment = asset_returns.get('NQ=F', 0.0)
  macro_adjustment = nasdaq_futures_sentiment * 0.15

  yesterday_nav = actual_nav_path[-1]
  gross_net_return = (
      (tier1_weight * t1_return)
      + (tier2_weight * t2_return)
      + macro_adjustment
      - daily_fee_drag
  )
  predicted_nav = yesterday_nav * (1.0 + gross_net_return)

  gia_val = gia_units * predicted_nav
  isa_val = isa_units * predicted_nav
  total_predicted_val = total_units * predicted_nav

  gia_prev = gia_units * yesterday_nav
  isa_prev = isa_units * yesterday_nav
  total_prev_val = total_units * yesterday_nav
  val_change = total_predicted_val - total_prev_val

  st.divider()
  st.subheader("Live Pre-Market Prediction")

  col1, col2 = st.columns(2)
  with col1:
    st.metric(
        "Estimated Live NAV",
        f"£{predicted_nav:.4f}",
        delta=f"{predicted_nav - yesterday_nav:.4f}",
    )
  with col2:
    st.metric("Prior Close NAV", f"£{yesterday_nav:.4f}")

  st.metric(
      "Total Portfolio Value",
      f"£{total_predicted_val:,.2f}",
      delta=f"£{val_change:+,.2f}",
  )
  st.write(
      f"**Optimized Weights** -> SOXX: {opt_soxx:.2f}, IGV: {opt_igv:.2f},"
      f" Core: {opt_core:.2f}"
  )

  st.divider()
  st.subheader("Daily Trade Decision Engine")

  if predicted_nav > yesterday_nav:
    st.success(
        f"DECISION: PROCEED (Estimated NAV £{predicted_nav:.4f} > Yesterday"
        f" £{yesterday_nav:.4f})"
    )
  else:
    st.warning(
        f"DECISION: HOLD / CANCEL (Estimated NAV £{predicted_nav:.4f} <= Yesterday"
        f" £{yesterday_nav:.4f})"
    )

else:
  st.error("Error: Insufficient live market data fetched for prediction.")
