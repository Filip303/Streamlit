import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

def get_portfolio_data(tickers, period, interval):
    portfolio_data = pd.DataFrame()
    info_dict = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if not df.empty:
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    portfolio_data[f"{ticker}_{col}"] = df[col]
                info_dict[ticker] = stock.info
        except Exception as e:
            st.warning(f"Error getting data for {ticker}: {e}")
    return portfolio_data, info_dict

def hierarchical_risk_parity(returns):
    try:
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.ffill().bfill()
        
        if returns.empty or returns.isna().all().all():
            raise ValueError("Insufficient data for calculation")
        
        corr = returns.corr()
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
        dist = np.nan_to_num(dist, nan=0.0)
        
        link = linkage(squareform(dist), method='ward')
        clustered_index = returns.columns[quasi_diag(link)]
        
        variances = returns[clustered_index].var()
        variances = variances.replace(0, np.finfo(float).eps)
        inv_var_weights = 1 / variances
        weights = inv_var_weights / inv_var_weights.sum()
        
        weights_series = pd.Series(weights.values, index=clustered_index)
        
        if weights_series.isna().any():
            raise ValueError("Error in weight calculation")
            
        return weights_series
        
    except Exception as e:
        st.error(f"Error in HRP: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

def quasi_diag(link):
    link = link.astype(int)
    num_items = link[-1, 3]
    sort_ix = []
    curr_index = link[-1, 0]
    
    while len(sort_ix) < num_items:
        if curr_index < num_items:
            sort_ix.append(curr_index)
        else:
            row = link[curr_index - num_items]
            sort_ix.extend([row[1], row[0]])
        if len(sort_ix) < num_items:
            curr_index = sort_ix.pop()
            
    return sort_ix

def get_benchmark_data(period, interval):
    try:
        benchmarks = ['SPY', 'URTH']
        benchmark_data = pd.DataFrame()
        
        for ticker in benchmarks:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if not df.empty:
                benchmark_data[f"{ticker}_Close"] = df['Close']
        
        return benchmark_data
    except Exception as e:
        st.error(f"Error getting benchmark data: {e}")
        return None

def calculate_technical_indicators(df, symbol):
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    close = df[f'{symbol}_Close']
    high = df[f'{symbol}_High']
    low = df[f'{symbol}_Low']
    volume = df[f'{symbol}_Volume']
    
    df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(high=high, low=low, close=close, volume=volume)
    df[f'{symbol}_EMA20'] = ta.trend.ema_indicator(close, window=20)
    df[f'{symbol}_EMA50'] = ta.trend.ema_indicator(close, window=50)
    df[f'{symbol}_SMA20'] = close.rolling(window=20).mean()
    df[f'{symbol}_SMA50'] = close.rolling(window=50).mean()
    
    df[f'{symbol}_RSI'] = ta.momentum.rsi(close)
    df[f'{symbol}_MACD'] = ta.trend.macd_diff(close)
    df[f'{symbol}_BB_upper'] = ta.volatility.bollinger_hband(close)
    df[f'{symbol}_BB_middle'] = ta.volatility.bollinger_mavg(close)
    df[f'{symbol}_BB_lower'] = ta.volatility.bollinger_lband(close)
    df[f'{symbol}_ADX'] = ta.trend.adx(high, low, close)
    df[f'{symbol}_OBV'] = ta.volume.on_balance_volume(close, volume)
    df[f'{symbol}_ATR'] = ta.volatility.average_true_range(high, low, close)
    
    return df.ffill().bfill()

def calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate):
    try:
        metrics = {}
        
        returns = portfolio_data.filter(like='Close').pct_change()
        returns.columns = [col.replace('_Close', '') for col in returns.columns]
        portfolio_return = (returns * weights).sum(axis=1)
        
        benchmark_data = get_benchmark_data(period, interval)
        benchmark_returns = benchmark_data.pct_change() if benchmark_data is not None else None
        
        for name, ret in [('Portfolio', portfolio_return)] + (
            [('SPY', benchmark_returns['SPY_Close']), ('URTH', benchmark_returns['URTH_Close'])] 
            if benchmark_returns is not None else []
        ):
            mean_return = ret.mean()
            std_return = ret.std() or 1e-6
            
            sharpe = np.sqrt(252) * (mean_return - risk_free_rate/252) / std_return
            
            downside_returns = ret[ret < 0]
            downside_std = downside_returns.std() or 1e-6
            sortino = np.sqrt(252) * (mean_return - risk_free_rate/252) / downside_std
            
            cum_returns = (1 + ret).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            calmar = abs(252 * mean_return / (max_drawdown or -1e-6))
            
            metrics[name] = {
                'Sharpe': np.clip(sharpe, -100, 100),
                'Sortino': np.clip(sortino, -100, 100),
                'Calmar': np.clip(calmar, -100, 100),
                'Max Drawdown': max_drawdown,
                'Returns': cum_returns
            }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error in metrics calculation: {e}")
        return None

# Main app
st.set_page_config(page_title="Trading Platform", layout="wide")

with st.sidebar:
    symbols_input = st.text_input("Symbols (comma separated)", "AAPL,MSFT,GOOGL")
    symbols = [s.strip() for s in symbols_input.split(",")]
    
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1d", "5d", "1wk", "1mo"], index=0)
    chart_type = st.selectbox("Chart Type", ['Candlestick', 'OHLC', 'Line'])
    use_log = st.checkbox("Logarithmic Scale", value=False)
    risk_free_rate = st.number_input("Annual Risk-Free Rate (%)", 0.0, 100.0, 2.0) / 100.0
    
    available_indicators = [
        'EMA20', 'EMA50', 'SMA20', 'SMA50', 'VWAP',
        'RSI', 'MACD', 'Bollinger Bands', 'ADX', 'OBV', 'ATR'
    ]
    selected_indicators = st.multiselect("Technical Indicators", available_indicators)

portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    weights = hierarchical_risk_parity(returns)
    
    tab1, tab2, tab3 = st.tabs(["Portfolio Analysis", "Performance Metrics", "Technical Analysis"])
    
    with tab1:
        st.subheader("Portfolio Composition (HRP)")
        weights_df = pd.DataFrame({'Asset': weights.index, 'Weight': weights.values * 100})
        
        if len(symbols) > 1:
            fig = go.Figure(data=[go.Pie(labels=weights_df['Asset'],
                                       values=weights_df['Weight'],
                                       textinfo='label+percent')])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please add more symbols in the sidebar to view portfolio composition")
        
        st.dataframe(weights_df.round(2))
    
    with tab2:
        metrics = calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate)
        if metrics:
            columns = ['Portfolio']
            if 'SPY' in metrics:
                columns.extend(['SPY', 'URTH'])
                
            metrics_df = pd.DataFrame({
                'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown'],
                **{col: [
                    f"{metrics[col]['Sharpe']:.2f}",
                    f"{metrics[col]['Sortino']:.2f}",
                    f"{metrics[col]['Calmar']:.2f}",
                    f"{metrics[col]['Max Drawdown']:.2%}"
                ] for col in columns}
            })
            
            st.dataframe(metrics_df)
            
            fig = go.Figure()
            for col in columns:
                fig.add_trace(go.Scatter(
                    x=metrics[col]['Returns'].index,
                    y=metrics[col]['Returns'],
                    name=col
                ))
            
            fig.update_layout(
                title="Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                height=500,
                yaxis_type='log' if use_log else 'linear'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        selected_symbol = st.selectbox("Select Asset", symbols)
        technical_data = calculate_technical_indicators(portfolio_data, selected_symbol)
        
        fig = go.Figure()
        
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(
                x=technical_data.index,
                open=technical_data[f'{selected_symbol}_Open'],
                high=technical_data[f'{selected_symbol}_High'],
                low=technical_data[f'{selected_symbol}_Low'],
                close=technical_data[f'{selected_symbol}_Close'],
                name=selected_symbol
            ))
        elif chart_type == 'OHLC':
            fig.add_trace(go.Ohlc(
                x=technical_data.index,
                open=technical_data[f'{selected_symbol}_Open'],
                high=technical_data[f'{selected_symbol}_High'],
                low=technical_data[f'{selected_symbol}_Low'],
                close=technical_data[f'{selected_symbol}_Close'],
                name=selected_symbol
            ))
        else:
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_Close'],
                name=selected_symbol
            ))
            
        for indicator in selected_indicators:
            if indicator in ['EMA20', 'EMA50', 'SMA20', 'SMA50', 'VWAP']:
                fig.add_trace(go.Scatter(
                    x=technical_data.index,
                    y=technical_data[f'{selected_symbol}_{indicator}'],
                    name=indicator,
                    line=dict(dash='dash')
                ))
            elif indicator == 'Bollinger Bands':
                for band in ['upper', 'middle', 'lower']:
                    fig.add_trace(go.Scatter(
                        x=technical_data.index,
                        y=technical_data[f'{selected_symbol}_BB_{band}'],
                        name=f'BB {band}',
                        line=dict(dash='dot')
                    ))
                    
        fig.update_layout(
            title=f"Technical Analysis - {selected_symbol}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            yaxis_type='log' if use_log else 'linear'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        for indicator in selected_indicators:
            if indicator in ['RSI', 'MACD', 'ADX', 'OBV', 'ATR']:
                indicator_fig = go.Figure()
                indicator_fig.add_trace(go.Scatter(
                    x=technical_data.index,
                    y=technical_data[f'{selected_symbol}_{indicator}'],
                    name=indicator
                ))
                
                indicator_fig.update_layout(
                    title=f"{indicator} - {selected_symbol}",
                    xaxis_title="Date",
                    yaxis_title=indicator,
                    height=300
                )
                
                if indicator == 'RSI':
                    indicator_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    indicator_fig.add_hline(y=30, line_dash="dash", line_color="green")
                
                st.plotly_chart(indicator_fig, use_container_width=True)
                
        st.subheader("Trading Panel")
        col1, col2 = st.columns(2)
        
        with col1:
            operation = st.radio("Operation Type", ["Buy", "Sell"])
            quantity = st.number_input("Quantity", min_value=1, value=1)
        
        with col2:
            price = st.number_input("Price",
                                  min_value=0.01,
                                  value=float(technical_data[f'{selected_symbol}_Close'].iloc[-1]),
                                  format="%.2f")
            
            total = price * quantity
            st.write(f"Operation Total: ${total:,.2f}")
        
        if st.button("Execute Order"):st.success(f"{operation} order executed: {quantity} {selected_symbol} at ${price:.2f}")
            st.write(f"Total: ${total:,.2f}")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Demo platform - Not for real trading")