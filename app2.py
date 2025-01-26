import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def get_portfolio_data(tickers, period, interval):
    portfolio_data = pd.DataFrame()
    info_dict = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if df.empty or df.isna().all().all():
                st.warning(f"No data for ticker {ticker}. Skipping.")
                continue
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                portfolio_data[f"{ticker}_{col}"] = df[col]
            info_dict[ticker] = stock.info
        except Exception as e:
            st.warning(f"Error getting data for {ticker}: {e}")
    return portfolio_data, info_dict

def quasi_diag(link):
    link = link.astype(int)
    sort_ix = []
    sort_ix.extend([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    for i in range(len(link) - 2, -1, -1):
        if link[i, 0] >= num_items:
            sort_ix.append(link[i, 1])
        elif link[i, 1] >= num_items:
            sort_ix.append(link[i, 0])
    return np.array([x for x in sort_ix if x < num_items])

def hierarchical_risk_parity(returns):
    try:
        returns = returns.dropna(axis=1, how='all')
        if returns.empty:
            raise ValueError("No sufficient data in returns.")

        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, 'single')
        sort_ix = quasi_diag(link)
        
        var = returns.var()
        weights = 1/var
        weights = weights/weights.sum()
        weights = pd.Series(weights, index=returns.columns)
        
        return weights

    except Exception as e:
        st.error(f"Error in HRP calculation: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

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
    
    df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(
        high=high, low=low, close=close, volume=volume)
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
    
    return df.fillna(method='ffill').fillna(method='bfill')

def calculate_var_cvar(returns, confidence_level=0.95):
    try:
        if isinstance(returns, pd.Series):
            returns = returns.dropna()
        
        if len(returns) < 2:
            return 0, 0
            
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
        cvar = returns_array[returns_array <= var].mean()
        
        return var, cvar if not np.isnan(cvar) else var
        
    except Exception as e:
        st.error(f"Error calculating VaR/CVaR: {e}")
        return 0, 0

def calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate):
    try:
        metrics = {}
        
        returns = portfolio_data.filter(like='Close').pct_change()
        returns.columns = [col.replace('_Close', '') for col in returns.columns]
        portfolio_return = (returns * weights).sum(axis=1)
        
        mean_return = portfolio_return.mean()
        std_return = portfolio_return.std()
        
        if std_return > 0:
            sharpe = np.sqrt(252) * (mean_return - risk_free_rate/252) / std_return
            
            downside_returns = portfolio_return[portfolio_return < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else std_return
            sortino = np.sqrt(252) * (mean_return - risk_free_rate/252) / downside_std
            
            cum_returns = (1 + portfolio_return).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            var, cvar = calculate_var_cvar(portfolio_return)
            
            metrics['Portfolio'] = {
                'Returns': cum_returns,
                'Sharpe': sharpe,
                'Sortino': sortino,
                'Max Drawdown': max_drawdown,
                'VaR': var,
                'CVaR': cvar
            }
            
            # Calculate benchmark metrics if available
            benchmark_data = get_benchmark_data(period, interval)
            if benchmark_data is not None:
                for bench in ['SPY', 'URTH']:
                    bench_returns = benchmark_data[f'{bench}_Close'].pct_change()
                    bench_mean = bench_returns.mean()
                    bench_std = bench_returns.std()
                    
                    if bench_std > 0:
                        bench_sharpe = np.sqrt(252) * (bench_mean - risk_free_rate/252) / bench_std
                        bench_downside = bench_returns[bench_returns < 0]
                        bench_downside_std = bench_downside.std() if not bench_downside.empty else bench_std
                        bench_sortino = np.sqrt(252) * (bench_mean - risk_free_rate/252) / bench_downside_std
                        
                        bench_cum_returns = (1 + bench_returns).cumprod()
                        bench_rolling_max = bench_cum_returns.expanding().max()
                        bench_drawdowns = (bench_cum_returns - bench_rolling_max) / bench_rolling_max
                        bench_max_drawdown = bench_drawdowns.min()
                        
                        bench_var, bench_cvar = calculate_var_cvar(bench_returns)
                        
                        metrics[bench] = {
                            'Returns': bench_cum_returns,
                            'Sharpe': bench_sharpe,
                            'Sortino': bench_sortino,
                            'Max Drawdown': bench_max_drawdown,
                            'VaR': bench_var,
                            'CVaR': bench_cvar
                        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error in metrics calculation: {e}")
        return None

# Configuración de la página
st.set_page_config(page_title="Trading Platform V2", layout="wide")

# Configuración de la barra lateral
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

# Obtener datos del portafolio
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
                
            metrics_data = {
                'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
            }
            
            for col in columns:
                metrics_data[col] = [
                    f"{metrics[col]['Sharpe']:.2f}",
                    f"{metrics[col]['Sortino']:.2f}",
                    f"{metrics[col]['Max Drawdown']:.2%}",
                    f"{metrics[col]['VaR']:.2%}",
                    f"{metrics[col]['CVaR']:.2%}"
                ]
            
            metrics_df = pd.DataFrame(metrics_data)
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
            yaxis_title="Date",
            yaxis_title="Price",
            height=600,
            yaxis_type='log' if use_log else 'linear'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráficos separados para indicadores
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

        # Panel de Trading
        st.subheader("Trading Panel")
        col1, col2 = st.columns(2)
        
        with col1:
            operation = st.radio("Operation Type", ["Buy", "Sell"])
            quantity = st.number_input("Quantity", min_value=1, value=1)
        
        with col2:
            price = st.number_input(
                "Price",
                min_value=0.01,
                value=float(technical_data[f'{selected_symbol}_Close'].iloc[-1]),
                format="%.2f"
            )
            
            total = price * quantity
            st.write(f"Total operation value: ${total:,.2f}")
        
        if st.button("Execute Order"):
            st.success(f"{operation} order executed: {quantity} {selected_symbol} at ${price:.2f}")

# Mensaje en la barra lateral
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Demo platform - Not for real trading")