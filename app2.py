import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
import statsmodels.api as sm
import ta

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

def calculate_har_volatility(returns, lags=[1, 5, 22]):
    """Calculate HAR volatility forecast"""
    # Calculate realized variance (squared returns)
    rv = returns ** 2
    
    # Calculate average RV for different horizons
    rv_daily = rv.rolling(window=lags[0]).mean()
    rv_weekly = rv.rolling(window=lags[1]).mean()
    rv_monthly = rv.rolling(window=lags[2]).mean()
    
    # Create lagged features
    X = pd.DataFrame({
        'daily': rv_daily.shift(1),
        'weekly': rv_weekly.shift(1),
        'monthly': rv_monthly.shift(1)
    }).fillna(method='bfill')
    
    y = rv
    
    # Simple OLS estimation
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # Forecast
    forecast = model.predict(X.iloc[-1:]).iloc[0]
    return np.sqrt(forecast)

def calculate_dynamic_levels(data, symbol, confidence_level=0.95):
    """Calculate dynamic stop loss and take profit using HAR volatility"""
    returns = pd.Series(np.log(data[f'{symbol}_Close']).diff().dropna())
    
    # Calculate HAR volatility forecast
    conditional_vol = calculate_har_volatility(returns)
    
    z_score = norm.ppf(confidence_level)
    current_price = data[f'{symbol}_Close'].iloc[-1]
    
    stop_loss = current_price * np.exp(-z_score * conditional_vol)
    take_profit = current_price * np.exp(z_score * conditional_vol)
    
    return stop_loss, take_profit, conditional_vol

def hierarchical_risk_parity(returns):
    try:
        returns = returns.dropna(axis=1, how='all')
        if returns.empty:
            raise ValueError("No sufficient data in returns.")

        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, 'single')
        
        var = returns.var()
        weights = 1 / var
        weights = weights / weights.sum()
        weights = pd.Series(weights, index=returns.columns)
        
        return weights
    except Exception as e:
        st.error(f"Error in HRP calculation: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

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
            sharpe = np.sqrt(252) * (mean_return - risk_free_rate / 252) / std_return
            
            downside_returns = portfolio_return[portfolio_return < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else std_return
            sortino = np.sqrt(252) * (mean_return - risk_free_rate / 252) / downside_std
            
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
            
            # Calculate benchmark metrics
            benchmark_data = get_benchmark_data(period, interval)
            if benchmark_data is not None:
                for bench in ['SPY', 'URTH']:
                    bench_returns = benchmark_data[f'{bench}_Close'].pct_change()
                    bench_mean = bench_returns.mean()
                    bench_std = bench_returns.std()
                    
                    if bench_std > 0:
                        bench_sharpe = np.sqrt(252) * (bench_mean - risk_free_rate / 252) / bench_std
                        bench_downside = bench_returns[bench_returns < 0]
                        bench_downside_std = bench_downside.std() if not bench_downside.empty else bench_std
                        bench_sortino = np.sqrt(252) * (bench_mean - risk_free_rate / 252) / bench_downside_std
                        
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
st.set_page_config(page_title="Trading Platform Pro V3", layout="wide")
st.title("📈 Trading Platform Pro V3")

# Sidebar
with st.sidebar:
    st.header("Configuración")
    symbols_input = st.text_input("Símbolos (separados por coma)", "AAPL,MSFT,GOOGL")
    symbols = [s.strip() for s in symbols_input.split(",")]
    
    period = st.selectbox(
        "Período",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )
    interval = st.selectbox(
        "Intervalo",
        options=["1d", "5d", "1wk", "1mo"],
        index=0
    )
    chart_type = st.selectbox("Tipo de Gráfico", ['Candlestick', 'OHLC', 'Line'])
    use_log = st.checkbox("Escala Logarítmica", value=False)
    confidence_level = st.slider("Nivel de Confianza (%)", 90, 99, 95) / 100
    risk_free_rate = st.number_input("Tasa Libre de Riesgo Anual (%)", 0.0, 100.0, 2.0) / 100.0
    
    available_indicators = [
        'EMA20', 'EMA50', 'SMA20', 'SMA50', 'VWAP',
        'RSI', 'MACD', 'Bollinger Bands', 'ADX', 'OBV', 'ATR'
    ]
    selected_indicators = st.multiselect("Indicadores Técnicos", available_indicators)

# Obtener datos
portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    weights = hierarchical_risk_parity(returns)
    
    tab1, tab2, tab3 = st.tabs(["Análisis de Cartera", "Análisis Técnico", "Panel de Trading"])
    
    with tab1:
        st.subheader("Composición de la Cartera (HRP)")
        weights_df = pd.DataFrame({'Activo': weights.index, 'Peso': weights.values * 100})
        
        if len(symbols) > 1:
            fig = go.Figure(data=[go.Pie(labels=weights_df['Activo'],
                                       values=weights_df['Peso'],
                                       textinfo='label+percent')])
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(weights_df.round(2))
        
        metrics = calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate)
        if metrics:
            st.subheader("Métricas de Rendimiento")
            cols = st.columns(len(metrics))
            
            for i, (name, metric) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(f"{name} Sharpe", f"{metric['Sharpe']:.2f}")
                    st.metric(f"{name} Sortino", f"{metric['Sortino']:.2f}")
                    st.metric(f"{name} Max Drawdown", f"{metric['Max Drawdown']:.2%}")
                    st.metric(f"{name} VaR", f"{metric['VaR']:.2%}")
                    st.metric(f"{name} CVaR", f"{metric['CVaR']:.2%}")
            
            fig = go.Figure()
            for name, metric in metrics.items():
                fig.add_trace(go.Scatter(
                    x=metric['Returns'].index,
                    y=metric['Returns'],
                    name=name
                ))
            
            fig.update_layout(
                title="Comparación de Rendimiento",
                xaxis_title="Fecha",
                yaxis_title="Retorno Acumulado",
                height=500,
                yaxis_type='log' if use_log else 'linear'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        selected_symbol = st.selectbox("Seleccionar Activo", symbols)
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
            title=f"Análisis Técnico - {selected_symbol}",
            xaxis_title="Fecha",
            yaxis_title="Precio",
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
                    xaxis_title="Fecha",
                    yaxis_title=indicator,
                    height=300
                )
                
                if indicator == 'RSI':
                    indicator_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    indicator_fig.add_hline(y=30, line_dash="dash", line_color="green")
                
                st.plotly_chart(indicator_fig, use_container_width=True)
    
    with tab3:
        selected_symbol = st.selectbox("Seleccionar Activo para Trading", symbols)
        stop_loss, take_profit, volatility = calculate_dynamic_levels(
            portfolio_data, selected_symbol, confidence_level)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            if chart_type == 'Candlestick':
                fig.add_trace(go.Candlestick(
                    x=portfolio_data.index,
                    open=portfolio_data[f'{selected_symbol}_Open'],
                    high=portfolio_data[f'{selected_symbol}_High'],
                    low=portfolio_data[f'{selected_symbol}_Low'],
                    close=portfolio_data[f'{selected_symbol}_Close'],
                    name=selected_symbol
                ))
            elif chart_type == 'OHLC':
                fig.add_trace(go.Ohlc(
                    x=portfolio_data.index,
                    open=portfolio_data[f'{selected_symbol}_Open'],
                    high=portfolio_data[f'{selected_symbol}_High'],
                    low=portfolio_data[f'{selected_symbol}_Low'],
                    close=portfolio_data[f'{selected_symbol}_Close'],
                    name=selected_symbol
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data[f'{selected_symbol}_Close'],
                    name=selected_symbol
                ))
            
            current_price = portfolio_data[f'{selected_symbol}_Close'].iloc[-1]
            
            fig.add_hline(y=stop_loss, line_color="red", line_dash="dash",
                         annotation_text="Stop Loss Dinámico")
            fig.add_hline(y=take_profit, line_color="green", line_dash="dash",
                         annotation_text="Take Profit Dinámico")
            
            fig.update_layout(
                title=f"Trading View - {selected_symbol}",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                height=600,
                yaxis_type='log' if use_log else 'linear'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Panel de Trading")
            
            if info_dict.get(selected_symbol):
                info = info_dict[selected_symbol]
                st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industria:** {info.get('industry', 'N/A')}")
            
            st.metric("Precio Actual", f"${current_price:.2f}")
            st.metric("Volatilidad Condicional", f"{volatility:.2%}")
            
            operation = st.radio("Tipo de Operación", ["Comprar", "Vender"])
            quantity = st.number_input("Cantidad", min_value=1, value=1)
            
            use_dynamic_levels = st.checkbox("Usar Niveles Dinámicos", value=True)
            
            if use_dynamic_levels:
                sl_price = stop_loss
                tp_price = take_profit
            else:
                sl_price = st.number_input("Stop Loss Manual", 
                                         value=float(stop_loss),
                                         step=0.01)
                tp_price = st.number_input("Take Profit Manual", 
                                         value=float(take_profit),
                                         step=0.01)
            
            st.metric("Stop Loss", f"${sl_price:.2f}", 
                     f"{(sl_price/current_price - 1):.2%}")
            st.metric("Take Profit", f"${tp_price:.2f}", 
                     f"{(tp_price/current_price - 1):.2%}")
            
            total = current_price * quantity
            st.write(f"Total de la operación: ${total:,.2f}")
            
            if st.button("Ejecutar Orden"):
                st.success(f"""
                Orden ejecutada:
                - {operation} {quantity} {selected_symbol} a ${current_price:.2f}
                - Stop Loss: ${sl_price:.2f} ({(sl_price/current_price - 1):.2%})
                - Take Profit: ${tp_price:.2f} ({(tp_price/current_price - 1):.2%})
                - Total: ${total:,.2f}
                """)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Características:
- Análisis de cartera con HRP
- Indicadores técnicos avanzados
- Stop Loss/Take Profit dinámicos usando VAR y HAR
- Optimización de cartera
- Múltiples tipos de gráficos
- Escala logarítmica opcional
""")
st.sidebar.warning("Plataforma de demostración - No usar para trading real.")
