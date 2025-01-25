import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# Translation dictionaries
translations = {
    'en': {
        'title': '📈 Advanced Trading Platform',
        'config': 'Configuration',
        'symbols': 'Symbols (comma separated)',
        'period': 'Period',
        'interval': 'Interval',
        'chart_type': 'Chart Type',
        'log_scale': 'Logarithmic Scale',
        'indicators': 'Technical Indicators',
        'analysis': 'Portfolio Analysis',
        'metrics': 'Performance Metrics',
        'technical': 'Technical Analysis',
        'buy': 'Buy',
        'sell': 'Sell',
        'hold': 'Hold',
        'quantity': 'Quantity',
        'price': 'Price',
        'select_asset': 'Select Asset',
        'current_signal': 'Current Signal for',
        'risk_free_rate': 'Annual Risk-Free Rate (%)'
    },
    'es': {
        'title': '📈 Plataforma de Trading Avanzada',
        'config': 'Configuración',
        'symbols': 'Símbolos (separados por coma)',
        'period': 'Período',
        'interval': 'Intervalo',
        'chart_type': 'Tipo de Gráfico',
        'log_scale': 'Escala Logarítmica',
        'indicators': 'Indicadores Técnicos',
        'analysis': 'Análisis de Cartera',
        'metrics': 'Métricas de Rendimiento',
        'technical': 'Análisis Técnico',
        'buy': 'Comprar',
        'sell': 'Vender',
        'hold': 'Mantener',
        'quantity': 'Cantidad',
        'price': 'Precio',
        'select_asset': 'Seleccionar Activo',
        'current_signal': 'Señal actual para',
        'risk_free_rate': 'Tasa Libre de Riesgo Anual (%)'
    }
}

@st.cache_data
def get_portfolio_data(tickers, period, interval):
    try:
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
                continue
        
        return portfolio_data, info_dict
    except Exception as e:
        st.error(f"Portfolio data error: {e}")
        return None, None

def calculate_metrics_with_benchmark(portfolio_data, weights, risk_free_rate):
    try:
        returns = portfolio_data.filter(like='Close').pct_change()
        returns.columns = [col.replace('_Close', '') for col in returns.columns]
        portfolio_return = (returns * weights).sum(axis=1)
        
        metrics = {}
        
        # Portfolio metrics
        mean_return = portfolio_return.mean()
        std_return = portfolio_return.std() or 1e-6
        
        sharpe = np.sqrt(252) * (mean_return - risk_free_rate/252) / std_return
        
        downside_returns = portfolio_return[portfolio_return < 0]
        downside_std = downside_returns.std() or 1e-6
        sortino = np.sqrt(252) * (mean_return - risk_free_rate/252) / downside_std
        
        cum_returns = (1 + portfolio_return).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        calmar = abs(252 * mean_return / (max_drawdown or -1e-6))
        
        metrics['Portfolio'] = {
            'Sharpe': np.clip(sharpe, -100, 100),
            'Sortino': np.clip(sortino, -100, 100),
            'Calmar': np.clip(calmar, -100, 100),
            'Max Drawdown': max_drawdown,
            'Returns': cum_returns
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error in metrics calculation: {e}")
        return None(df, symbol):
    try:
        df = df.copy()
        df = df.sort_index()
        
        close = df[f'{symbol}_Close']
        high = df[f'{symbol}_High']
        low = df[f'{symbol}_Low']
        volume = df[f'{symbol}_Volume']
        
        # Basic indicators
        df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(
            high=high, low=low, close=close, volume=volume, window=14)
        df[f'{symbol}_EMA20'] = ta.trend.ema_indicator(close, window=20)
        df[f'{symbol}_EMA50'] = ta.trend.ema_indicator(close, window=50)
        df[f'{symbol}_SMA20'] = close.rolling(window=20).mean()
        df[f'{symbol}_SMA50'] = close.rolling(window=50).mean()
        
        # Additional indicators
        df[f'{symbol}_RSI'] = ta.momentum.rsi(close, window=14)
        df[f'{symbol}_MACD'] = ta.trend.macd_diff(close)
        df[f'{symbol}_BB_upper'], df[f'{symbol}_BB_middle'], df[f'{symbol}_BB_lower'] = ta.volatility.bollinger_bands(close)
        df[f'{symbol}_ADX'] = ta.trend.adx(high, low, close)
        df[f'{symbol}_OBV'] = ta.volume.on_balance_volume(close, volume)
        df[f'{symbol}_ATR'] = ta.volatility.average_true_range(high, low, close)
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    except Exception as e:
        st.error(f"Technical indicators error: {e}")
        return df

# Page configuration
st.set_page_config(page_title="Trading Platform V2", layout="wide")

# Language selection
language = st.sidebar.selectbox("Language / Idioma", ['en', 'es'])
t = translations[language]

st.title(t['title'])

with st.sidebar:
    st.header(t['config'])
    symbols_input = st.text_input(t['symbols'], value="AAPL,MSFT,GOOGL")
    symbols = [s.strip() for s in symbols_input.split(",")]
    
    period = st.selectbox(
        t['period'],
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )
    
    interval = st.selectbox(
        t['interval'],
        options=["1d", "5d", "1wk", "1mo"],
        index=0
    )
    
    chart_type = st.selectbox(
        t['chart_type'],
        options=['Candlestick', 'OHLC', 'Line']
    )
    
    use_log = st.checkbox(t['log_scale'], value=False)
    
    risk_free_rate = st.number_input(t['risk_free_rate'], min_value=0.0, max_value=100.0, value=2.0) / 100.0
    
    available_indicators = [
        'EMA20', 'EMA50', 'SMA20', 'SMA50', 'VWAP',
        'RSI', 'MACD', 'Bollinger Bands', 'ADX', 'OBV', 'ATR'
    ]
    selected_indicators = st.multiselect(t['indicators'], available_indicators)

portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    
    try:
        weights = hierarchical_risk_parity(returns)
    except Exception as e:
        st.error(f"Error in HRP calculation: {e}")
        weights = pd.Series({symbol: 1/len(symbols) for symbol in symbols})
    
    tab1, tab2, tab3 = st.tabs([t['analysis'], t['metrics'], t['technical']])
    
    with tab3:
        selected_symbol = st.selectbox(t['select_asset'], symbols)
        technical_data = calculate_technical_indicators(portfolio_data.copy(), selected_symbol)
        
        # Main price chart
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
        else:  # Line chart
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_Close'],
                name=selected_symbol
            ))
        
        # Add selected indicators
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
            title=f"{selected_symbol} - {t['technical']}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            yaxis_type='log' if use_log else 'linear'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Separate charts for other indicators
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
        
        # Trading signals
        st.subheader(f"{t['current_signal']} {selected_symbol}")
        signal = 'HOLD'
        if technical_data[f'{selected_symbol}_EMA20'].iloc[-1] > technical_data[f'{selected_symbol}_EMA50'].iloc[-1]:
            signal = 'BUY'
        elif technical_data[f'{selected_symbol}_EMA20'].iloc[-1] < technical_data[f'{selected_symbol}_EMA50'].iloc[-1]:
            signal = 'SELL'
            
        signal_translation = {'BUY': t['buy'], 'SELL': t['sell'], 'HOLD': t['hold']}
        st.info(f"{t['current_signal']} {selected_symbol}: {signal_translation[signal]}")
        


if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Demo platform - Not for real trading / Plataforma de demostración - No usar para trading real")