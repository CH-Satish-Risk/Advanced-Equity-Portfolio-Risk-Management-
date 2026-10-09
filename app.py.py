# ========================================== 
# ADVANCED EQUITY PORTFOLIO RISK DASHBOARD
# ========================================== 

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from arch import arch_model
import scipy.stats as stats

# Page Configuration
st.set_page_config(
    page_title="Portfolio Risk Analytics Platform",
    page_icon="📈",
    layout="wide"
)

st.title("🛡️ Advanced Equity Portfolio Risk Management Platform")
st.markdown("Interactive Risk Analytics: Historical Simulation, EWMA, GARCH, GJR-GARCH, Backtesting & Stress Testing")

# ------------------------------------------
# SIDEBAR: GLOBAL INPUTS
# ------------------------------------------
st.sidebar.header("Portfolio Parameters")

default_tickers = "AAPL, MSFT, NVDA, JPM, XOM, AMZN, META, GOOGL, TSLA, UNH"
ticker_input = st.sidebar.text_input("Asset Tickers (Comma-separated)", default_tickers)
tickers = [t.strip().upper() for t in ticker_input.split(',')]

initial_capital = st.sidebar.number_input("Initial Portfolio Value ($)", value=1000000.0, step=100000.0)

# Date range selection
lookback_years = st.sidebar.slider("Historical Data (Years)", min_value=1, max_value=5, value=3)
end_date = datetime.date.today().strftime('%Y-%m-%d')
start_date = (datetime.date.today() - datetime.timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')

# ------------------------------------------
# DATA LOADING & CACHING
# ------------------------------------------
@st.cache_data
def load_data(ticker_list, start, end):
    # Added auto_adjust=False to prevent KeyError: 'Adj Close'
    raw_data = yf.download(ticker_list, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(raw_data.columns, pd.MultiIndex):
        prices = raw_data['Adj Close']
    else:
        prices = raw_data[['Adj Close']]
    prices = prices.dropna(how='all').ffill().dropna()
    return prices

with st.spinner("Fetching market data from Yahoo Finance..."):
    stock_prices = load_data(tickers, start_date, end_date)

if stock_prices.empty:
    st.error("No data fetched. Please check ticker symbols and date range.")
    st.stop()

# Equal weights default
weights = np.array([1.0 / len(tickers)] * len(tickers))

# Log returns & Portfolio construction
log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
portfolio_log_returns = log_returns.dot(weights)
portfolio_simple_returns = np.exp(portfolio_log_returns) - 1
portfolio_pnl = initial_capital * portfolio_simple_returns
portfolio_value = initial_capital * (1 + portfolio_simple_returns).cumprod()
portfolio_value.iloc[0] = initial_capital

# ------------------------------------------
# TAB NAVIGATION
# ------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio",
    "📉 VaR & Expected Shortfall",
    "⚡ Volatility Scaling",
    "🔍 Backtesting",
    "🚨 Stress Testing"
])

# --- TAB 1: PORTFOLIO PAGE ---
with tab1:
    st.header("Portfolio Construction & Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset Allocation (Equal-Weighted)")
        weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': weights})
        st.dataframe(weights_df, use_container_width=True)
    with col2:
        st.metric("Initial Capital", f"${initial_capital:,.2f}")
        st.metric("Final Portfolio Value", f"${portfolio_value.iloc[-1]:,.2f}")
        total_ret = (portfolio_value.iloc[-1] / initial_capital - 1) * 100
        st.metric("Total Cumulative Return", f"{total_ret:.2f}%")

    # Plot Portfolio Value
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Portfolio Value', line=dict(color='forestgreen', width=2)))
    fig_val.update_layout(title="Portfolio Value Over Time ($)", xaxis_title="Date", yaxis_title="Value ($)", template="plotly_white")
    st.plotly_chart(fig_val, use_container_width=True)

# --- TAB 2: VaR & EXPECTED SHORTFALL ---
with tab2:
    st.header("Historical Simulation VaR & Expected Shortfall")

    col1, col2 = st.columns(2)
    with col1:
        hist_window = st.selectbox("Historical Window (Days)", [250, 500, 750], index=1)
    with col2:
        conf_level = st.selectbox("Confidence Level", [0.95, 0.99, 0.995], index=1)

    alpha = 1.0 - conf_level
    roll_var = portfolio_pnl.rolling(window=hist_window).apply(lambda x: -np.percentile(x, alpha * 100), raw=True)
    roll_es = portfolio_pnl.rolling(window=hist_window).apply(lambda x: -np.mean(x[x <= -np.percentile(x, alpha * 100)]), raw=True)

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(x=portfolio_pnl.index, y=-portfolio_pnl, mode='lines', name='Daily Loss ($)', line=dict(color='gray', width=1, dash='dot')))
    fig_var.add_trace(go.Scatter(x=roll_var.index, y=roll_var, mode='lines', name=f'Historical VaR ({int(conf_level*100)}%)', line=dict(color='crimson', width=2)))
    fig_var.add_trace(go.Scatter(x=roll_es.index, y=roll_es, mode='lines', name=f'Expected Shortfall ({int(conf_level*100)}%)', line=dict(color='darkred', width=2, dash='dash')))
    fig_var.update_layout(title=f"Rolling Historical VaR & Expected Shortfall ({hist_window}-Day Window)", xaxis_title="Date", yaxis_title="Loss ($)", template="plotly_white")
    st.plotly_chart(fig_var, use_container_width=True)

# --- TAB 3: VOLATILITY SCALING ---
with tab3:
    st.header("Volatility Scaling: EWMA, GARCH, & GJR-GARCH")

    with st.spinner("Fitting GARCH and GJR-GARCH volatility models..."):
        garch_returns = portfolio_simple_returns.dropna() * 100.0

        # EWMA
        lam = 0.94
        ewma_var_s = pd.Series(index=garch_returns.index, dtype=float)
        ewma_var_s.iloc[0] = garch_returns.iloc[:20].var()
        for i in range(1, len(garch_returns)):
            ewma_var_s.iloc[i] = lam * ewma_var_s.iloc[i-1] + (1 - lam) * (garch_returns.iloc[i-1] ** 2)
        ewma_vol = np.sqrt(ewma_var_s) / 100.0

        # GARCH(1,1)
        gm = arch_model(garch_returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
        gres = gm.fit(disp='off')
        garch_vol = gres.conditional_volatility / 100.0

        # GJR-GARCH(1,1,1)
        gjrm = arch_model(garch_returns, vol='Garch', p=1, o=1, q=1, dist='Normal')
        gjrres = gjrm.fit(disp='off')
        gjr_vol = gjrres.conditional_volatility / 100.0

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol*100, mode='lines', name='EWMA Volatility', line=dict(color='orange')))
    fig_vol.add_trace(go.Scatter(x=garch_vol.index, y=garch_vol*100, mode='lines', name='GARCH(1,1) Volatility', line=dict(color='green')))
    fig_vol.add_trace(go.Scatter(x=gjr_vol.index, y=gjr_vol*100, mode='lines', name='GJR-GARCH Volatility', line=dict(color='purple', width=2)))
    fig_vol.update_layout(title="Conditional Volatility Comparison (%)", xaxis_title="Date", yaxis_title="Daily Volatility (%)", template="plotly_white")
    st.plotly_chart(fig_vol, use_container_width=True)

# --- TAB 4: BACKTESTING ---
with tab4:
    st.header("VaR Backtesting & Basel Traffic Light Test")

    # Compute GJR-GARCH Scaled VaR for backtesting
    window = 500
    gjr_scaled_var = pd.Series(index=garch_returns.index, dtype=float)
    for i in range(window, len(garch_returns)):
        w_ret = portfolio_simple_returns.iloc[i - window : i]
        w_vol = gjr_vol.iloc[i - window : i]
        cur_vol = gjr_vol.iloc[i]
        scaled = w_ret * (cur_vol / w_vol)
        gjr_scaled_var.iloc[i] = -np.percentile(scaled, 0.01) * initial_capital

    bt_df = pd.DataFrame({'Actual_Loss': -portfolio_pnl, 'VaR': gjr_scaled_var}).dropna()
    bt_df['Exception'] = bt_df['Actual_Loss'] > bt_df['VaR']

    total_obs = len(bt_df)
    num_exc = bt_df['Exception'].sum()
    expected_exc = total_obs * 0.01

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", total_obs)
    col2.metric("Observed Exceptions", num_exc)
    col3.metric("Expected Exceptions", f"{expected_exc:.1f}")

    # Basel Zone determination
    scaled_exc = num_exc * (250.0 / total_obs)
    if scaled_exc <= 4:
        zone = "Green Zone (Pass)"
    elif scaled_exc < 10:
        zone = "Yellow Zone (Review)"
    else:
        zone = "Red Zone (Fail)"
    st.info(f"**Basel Traffic Light Classification:** {zone}")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Actual_Loss'], mode='lines', name='Actual Loss', line=dict(color='gray', width=1)))
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['VaR'], mode='lines', name='GJR-GARCH VaR (99%)', line=dict(color='purple', width=2)))
    exc_pts = bt_df[bt_df['Exception']]
    fig_bt.add_trace(go.Scatter(x=exc_pts.index, y=exc_pts['Actual_Loss'], mode='markers', name='Exceptions', marker=dict(color='red', size=8)))
    fig_bt.update_layout(title="Backtesting Exceptions Timeline", xaxis_title="Date", yaxis_title="Dollar Loss ($)", template="plotly_white")
    st.plotly_chart(fig_bt, use_container_width=True)

# --- TAB 5: STRESS TESTING ---
with tab5:
    st.header("Stress Testing & Scenario Analysis")

    scenarios = {
        'COVID Crash (2020)': portfolio_pnl.loc['2020-02-15':'2020-03-23'].sum(),
        'Inflation Shock (2022)': portfolio_pnl.loc['2022-01-03':'2022-06-30'].sum(),
        'Banking Stress (2023)': portfolio_pnl.loc['2023-03-01':'2023-03-31'].sum(),
        'Market Down 5% (Hypothetical)': -0.05 * portfolio_value.iloc[-1],
        'Market Down 10% (Hypothetical)': -0.10 * portfolio_value.iloc[-1],
        'Market Down 20% (Crash)': -0.20 * portfolio_value.iloc[-1]
    }

    stress_df = pd.DataFrame(list(scenarios.items()), columns=['Scenario', 'Estimated Loss ($)'])
    stress_df['Estimated Loss ($)'] = stress_df['Estimated Loss ($)'].round(2)
    stress_df = stress_df.sort_values(by='Estimated Loss ($)', ascending=True)

    st.dataframe(stress_df, use_container_width=True)

    fig_stress = go.Figure(go.Bar(
        x=stress_df['Estimated Loss ($)'],
        y=stress_df['Scenario'],
        orientation='h',
        marker_color='crimson'
    ))
    fig_stress.update_layout(title="Stress Testing Scenario Losses ($)", xaxis_title="Loss ($)", yaxis_title="Scenario", template="plotly_white")
    st.plotly_chart(fig_stress, use_container_width=True)
