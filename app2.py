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

# Función para obtener datos del portafolio
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
            st.warning(f"Error al obtener datos para {ticker}: {e}")
    return portfolio_data, info_dict

# Función para obtener datos de benchmarks
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
        st.error(f"Error al obtener datos de benchmark: {e}")
        return None

# Función para calcular la volatilidad HAR
def calculate_har_volatility(returns, lags=[1, 5, 22], scale_factor=2.5):
    rv = returns ** 2
    rv_daily = rv.rolling(window=lags[0]).mean()
    rv_weekly = rv.rolling(window=lags[1]).mean()
    rv_monthly = rv.rolling(window=lags[2]).mean()
    
    X = pd.DataFrame({
        'daily': rv_daily.shift(1),
        'weekly': rv_weekly.shift(1),
        'monthly': rv_monthly.shift(1)
    }).fillna(method='bfill')
    
    y = rv
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    forecast = model.predict(X.iloc[-1:]).iloc[0]
    return np.sqrt(forecast) * scale_factor

# Función para calcular indicadores Ichimoku
def calculate_ichimoku(df, symbol):
    high = df[f'{symbol}_High']
    low = df[f'{symbol}_Low']
    close = df[f'{symbol}_Close']
    
    nine_period_high = high.rolling(window=9).max()
    nine_period_low = low.rolling(window=9).min()
    df[f'{symbol}_tenkan_sen'] = (nine_period_high + nine_period_low) / 2
    
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    df[f'{symbol}_kijun_sen'] = (period26_high + period26_low) / 2
    
    df[f'{symbol}_senkou_span_a'] = ((df[f'{symbol}_tenkan_sen'] + 
        df[f'{symbol}_kijun_sen']) / 2).shift(26)
    
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    df[f'{symbol}_senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
    
    df[f'{symbol}_chikou_span'] = close.shift(-26)
    
    return df

# Función para calcular HRP (Hierarchical Risk Parity)
def hierarchical_risk_parity(returns):
    try:
        returns = returns.dropna(axis=1, how='all')
        if returns.empty:
            raise ValueError("No hay suficientes datos en los retornos.")

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
        st.error(f"Error en cálculo HRP: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

# Función para calcular indicadores técnicos
def calculate_technical_indicators(df, symbol):
    if df is None or df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    close = df[f'{symbol}_Close']
    high = df[f'{symbol}_High']
    low = df[f'{symbol}_Low']
    volume = df[f'{symbol}_Volume']
    
    # Indicadores básicos
    df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(
        high=high, low=low, close=close, volume=volume)
    df[f'{symbol}_EMA20'] = ta.trend.ema_indicator(close, window=20)
    df[f'{symbol}_EMA50'] = ta.trend.ema_indicator(close, window=50)
    df[f'{symbol}_SMA20'] = close.rolling(window=20).mean()
    df[f'{symbol}_SMA50'] = close.rolling(window=50).mean()
    
    # Indicadores de momentum
    df[f'{symbol}_RSI'] = ta.momentum.rsi(close)
    df[f'{symbol}_MACD'] = ta.trend.macd_diff(close)
    df[f'{symbol}_MACD_signal'] = ta.trend.macd_signal(close)
    df[f'{symbol}_MACD_line'] = ta.trend.macd(close)
    df[f'{symbol}_Stoch_RSI'] = ta.momentum.stochrsi(close)
    df[f'{symbol}_MFI'] = ta.volume.money_flow_index(high, low, close, volume)
    df[f'{symbol}_TSI'] = ta.momentum.tsi(close)
    
    # Indicadores de tendencia
    df[f'{symbol}_ADX'] = ta.trend.adx(high, low, close)
    df[f'{symbol}_CCI'] = ta.trend.cci(high, low, close)
    df[f'{symbol}_DPO'] = ta.trend.dpo(close)
    df[f'{symbol}_TRIX'] = ta.trend.trix(close)
    
    # Indicadores de volatilidad
    df[f'{symbol}_BB_upper'] = ta.volatility.bollinger_hband(close)
    df[f'{symbol}_BB_middle'] = ta.volatility.bollinger_mavg(close)
    df[f'{symbol}_BB_lower'] = ta.volatility.bollinger_lband(close)
    df[f'{symbol}_ATR'] = ta.volatility.average_true_range(high, low, close)
    df[f'{symbol}_KC_upper'] = ta.volatility.keltner_channel_hband(high, low, close)
    df[f'{symbol}_KC_lower'] = ta.volatility.keltner_channel_lband(high, low, close)
    
    # Indicadores de volumen
    df[f'{symbol}_OBV'] = ta.volume.on_balance_volume(close, volume)
    df[f'{symbol}_Force_Index'] = ta.volume.force_index(close, volume)
    df[f'{symbol}_EOM'] = ta.volume.ease_of_movement(high, low, volume)
    df[f'{symbol}_Volume_SMA'] = volume.rolling(window=20).mean()
    
    # Añadir indicadores Ichimoku
    df = calculate_ichimoku(df, symbol)
    
    return df.fillna(method='ffill').fillna(method='bfill')

# Función para calcular niveles dinámicos (Stop Loss y Take Profit)
def calculate_dynamic_levels(data, symbol, confidence_level=0.95, risk_multiplier=3):
    returns = pd.Series(np.log(data[f'{symbol}_Close']).diff().dropna())
    conditional_vol = calculate_har_volatility(returns)
    z_score = norm.ppf(confidence_level)
    current_price = data[f'{symbol}_Close'].iloc[-1]
    
    stop_loss = current_price * np.exp(-z_score * conditional_vol)
    risk = current_price - stop_loss
    take_profit = current_price + (risk * risk_multiplier)
    
    return stop_loss, take_profit, conditional_vol

# Función para calcular VaR y CVaR
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
        st.error(f"Error en cálculo VaR/CVaR: {e}")
        return 0, 0

# Función para calcular métricas del portafolio
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
        st.error(f"Error en cálculo de métricas: {e}")
        return None

# Función para graficar indicadores
def plot_indicators(fig, technical_data, selected_symbol, selected_indicators):
    for indicator in selected_indicators:
        if indicator == 'Ichimoku':
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_tenkan_sen'],
                name='Tenkan-sen',
                line=dict(color='blue', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_kijun_sen'],
                name='Kijun-sen',
                line=dict(color='red', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_senkou_span_a'],
                name='Senkou Span A',
                fill=None,
                line=dict(color='rgba(76,175,80,0.5)')
            ))
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_senkou_span_b'],
                name='Senkou Span B',
                fill='tonexty',
                line=dict(color='rgba(255,152,0,0.5)')
            ))
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_chikou_span'],
                name='Chikou Span',
                line=dict(color='purple')
            ))
        elif indicator in ['EMA20', 'EMA50', 'SMA20', 'SMA50', 'VWAP']:
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
        elif indicator == 'Keltner Channels':
            for band in ['upper', 'lower']:
                fig.add_trace(go.Scatter(
                    x=technical_data.index,
                    y=technical_data[f'{selected_symbol}_KC_{band}'],
                    name=f'KC {band}',
                    line=dict(dash='dot')
                ))
    return fig

# Función para crear subgráficos de indicadores
def create_indicator_subplot(technical_data, selected_symbol, indicator):
    fig = go.Figure()
    
    if indicator in ['RSI', 'Stoch RSI', 'MFI']:
        y_data = technical_data[f'{selected_symbol}_{indicator.replace(" ", "_")}']
        fig.add_trace(go.Scatter(x=technical_data.index, y=y_data, name=indicator))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    elif indicator == 'MACD':
        fig.add_trace(go.Scatter(
            x=technical_data.index,
            y=technical_data[f'{selected_symbol}_MACD_line'],
            name='MACD Line'
        ))
        fig.add_trace(go.Scatter(
            x=technical_data.index,
            y=technical_data[f'{selected_symbol}_MACD_signal'],
            name='Signal Line'
        ))
        fig.add_trace(go.Bar(
            x=technical_data.index,
            y=technical_data[f'{selected_symbol}_MACD'],
            name='MACD Histogram'
        ))
    else:
        y_data = technical_data[f'{selected_symbol}_{indicator.replace(" ", "_")}']
        fig.add_trace(go.Scatter(x=technical_data.index, y=y_data, name=indicator))
    
    fig.update_layout(
        title=f"{indicator} - {selected_symbol}",
        xaxis_title="Fecha",
        yaxis_title=indicator,
        height=300,
        yaxis_type='log'
    )
    
    return fig

# Configuración de la página
st.set_page_config(page_title="Trading Platform Pro V5", layout="wide")
st.title("📈 Trading Platform Pro V5")

# Sidebar
with st.sidebar:
    st.header("Configuración")
    symbols_input = st.text_input("Símbolos (separados por coma)", "AAPL,MSFT,GOOGL")
    symbols = [s.strip() for s in symbols_input.split(",")]
    
    period = st.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Intervalo", ["1d", "5d", "1wk", "1mo"], index=0)
    chart_type = st.selectbox("Tipo de Gráfico", ['Candlestick', 'OHLC', 'Line'])
    confidence_level = st.slider("Nivel de Confianza (%)", 90, 99, 95) / 100
    risk_free_rate = st.number_input("Tasa Libre de Riesgo Anual (%)", 0.0, 100.0, 2.0) / 100.0
    
    available_indicators = [
        'EMA20', 'EMA50', 'SMA20', 'SMA50', 'VWAP',
        'RSI', 'Stoch RSI', 'MACD', 'MFI', 'TSI',
        'Bollinger Bands', 'Keltner Channels', 'Ichimoku',
        'ADX', 'CCI', 'DPO', 'TRIX',
        'OBV', 'Force Index', 'EOM', 'Volume SMA'
    ]
    selected_indicators = st.multiselect("Indicadores Técnicos", available_indicators)

# Obtener datos
portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    weights = hierarchical_risk_parity(returns)
    metrics = calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate)
    
    # Panel de Métricas
    with st.expander("📊 Panel de Métricas", expanded=True):
        if metrics:
            metrics_df = pd.DataFrame(columns=['Métrica', 'Portfolio', 'SPY', 'URTH'])
            metrics_df['Métrica'] = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'VaR', 'CVaR']
            
            for name in ['Portfolio', 'SPY', 'URTH']:
                if name in metrics:
                    metrics_df[name] = [
                        f"{metrics[name]['Sharpe']:.2f}",
                        f"{metrics[name]['Sortino']:.2f}",
                        f"{metrics[name]['Max Drawdown']:.2%}",
                        f"{metrics[name]['VaR']:.2%}",
                        f"{metrics[name]['CVaR']:.2%}"
                    ]
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    tab1, tab2 = st.tabs(["Análisis de Cartera", "Análisis Técnico"])
    
    with tab1:
        st.subheader("Composición de la Cartera (HRP)")
        weights_df = pd.DataFrame({'Activo': weights.index, 'Peso': weights.values * 100})
        
        if len(symbols) > 1:
            fig = go.Figure(data=[go.Pie(labels=weights_df['Activo'],
                                       values=weights_df['Peso'],
                                       textinfo='label+percent')])
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(weights_df.round(2))
        
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
            yaxis_type='log'
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
        
        plot_indicators(fig, technical_data, selected_symbol, selected_indicators)
        
        fig.update_layout(
            title=f"Análisis Técnico - {selected_symbol}",
            xaxis_title="Fecha",
            yaxis_title="Precio",
            height=600,
            yaxis_type='log'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráficos separados para indicadores
        for indicator in selected_indicators:
            if indicator in ['RSI', 'Stoch RSI', 'MACD', 'MFI', 'TSI', 'ADX', 'CCI', 'DPO', 'TRIX', 'OBV', 'Force Index', 'EOM']:
                indicator_fig = create_indicator_subplot(technical_data, selected_symbol, indicator)
                st.plotly_chart(indicator_fig, use_container_width=True)
    
    # Panel de Trading
    with st.expander("💹 Panel de Trading", expanded=True):
        trading_symbol_input = st.text_input("Símbolo para Trading", "AAPL", key='trading_symbol_input')
        selected_symbol = trading_symbol_input.strip()
        
        try:
            # Verificar si el símbolo existe
            stock = yf.Ticker(selected_symbol)
            info = stock.info
            if not info:
                st.error(f"Símbolo {selected_symbol} no encontrado")
                st.stop()  # Detener la ejecución si no se encuentra el símbolo
        except Exception as e:
            st.error(f"Error al verificar símbolo: {e}")
            st.stop()  # Detener la ejecución si hay un error

        risk_multiplier = st.slider("Multiplicador de Riesgo para Take Profit", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
        
        stop_loss, take_profit, volatility = calculate_dynamic_levels(
            portfolio_data, selected_symbol, confidence_level, risk_multiplier)
        
        # Obtener datos específicos para trading
        trading_data = yf.Ticker(selected_symbol).history(period=period, interval=interval)
        if trading_data.empty:
            st.error("No se pudieron obtener datos para el símbolo seleccionado")
            st.stop()  # Detener la ejecución si no hay datos

        current_price = trading_data['Close'].iloc[-1]
        trading_data = trading_data.rename(columns={col: f"{selected_symbol}_{col}" for col in trading_data.columns})
        
        col1, col2 = st.columns([7, 3])
        
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
            else:
                fig.add_trace(go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data[f'{selected_symbol}_Close'],
                    name=selected_symbol
                ))

            # Añadir líneas de stop loss y take profit como traces
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=[stop_loss] * len(portfolio_data.index),
                mode='lines',
                name='Stop Loss',
                line=dict(color='red', dash='dash'),
                showlegend=True
            ))

            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=[take_profit] * len(portfolio_data.index),
                mode='lines',
                name='Take Profit',
                line=dict(color='green', dash='dash'),
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"Trading View - {selected_symbol}",
                xaxis_title="Fecha",
                yaxis_title="Precio",
                height=600,
                yaxis_type='log'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if info_dict.get(selected_symbol):
                info = info_dict[selected_symbol]
                st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industria:** {info.get('industry', 'N/A')}")
            
            st.metric("Precio Actual", f"${current_price:.2f}")
            st.metric("Volatilidad", f"{volatility:.2%}")
            st.metric("Stop Loss", f"${stop_loss:.2f}", 
                     f"{(stop_loss/current_price - 1):.2%}")
            st.metric("Take Profit", f"${take_profit:.2f}", 
                     f"{(take_profit/current_price - 1):.2%}")
            
            risk_amount = current_price - stop_loss
            reward_amount = take_profit - current_price
            risk_reward_ratio = reward_amount / risk_amount if risk_amount != 0 else float('inf')
            
            st.metric("Ratio Riesgo/Beneficio", f"{risk_reward_ratio:.2f}")
            
            quantity = st.number_input("Cantidad", min_value=1, value=1)
            total = current_price * quantity
            st.write(f"Total de la operación: ${total:,.2f}")
            
            risk_total = (current_price - stop_loss) * quantity
            reward_total = (take_profit - current_price) * quantity
            
            st.write(f"Riesgo máximo: ${risk_total:.2f}")
            st.write(f"Beneficio objetivo: ${reward_total:.2f}")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Características:
- Análisis de cartera con HRP
- Indicadores técnicos avanzados
- Stop Loss/Take Profit dinámicos con HAR
- Ratio de riesgo/beneficio ajustable
- Múltiples tipos de gráficos
- Escala logarítmica en todos los gráficos
""")
st.sidebar.warning("Plataforma de demostración - No usar para trading real.")
