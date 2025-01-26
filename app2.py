# ==================================================
# Importaciones y Configuración Inicial
# ==================================================
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
import requests
import fredapi

# API Keys
NEWS_API_KEY = "a00c1a624f854f3c9f48a167ed72eff1" 
ALPHA_VANTAGE_KEY = "9M00TPMNCN2ZW1G5"
FRED_API_KEY = "8617ec24219966a9191eb6a9d9d9fd24"

# Initialize FRED client
fred = fredapi.Fred(api_key=FRED_API_KEY)

# ==================================================
# Funciones de API
# ==================================================

# Función para obtener noticias por categoría
def get_news_by_category(category, page_size=10):
   queries = {
       "financial": "financial OR market OR stock",
       "macro": "macroeconomic OR economy OR gdp OR inflation",
       "political": "politics OR government OR regulation",
       "corporate": "earnings OR company OR corporate",
       "commodities": "commodities OR gold OR oil OR metals"
   }
   
   params = {
       "q": queries.get(category, ""),
       "apiKey": NEWS_API_KEY,
       "language": "en",
       "pageSize": page_size,
       "sortBy": "publishedAt"
   }
   
   try:
       response = requests.get("https://newsapi.org/v2/everything", params=params)
       return response.json()["articles"] if response.status_code == 200 else []
   except Exception as e:
       st.error(f"Error en API de noticias: {e}")
       return []

# Función para obtener datos fundamentales de un ticker
def get_financial_trends(ticker):
    """Obtener tendencias financieras históricas"""
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_KEY
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'annualReports' in data:
                reports = data['annualReports']
                return {
                    'dates': [r['fiscalDateEnding'] for r in reports],
                    'revenues': [float(r['totalRevenue']) for r in reports],
                    'net_income': [float(r['netIncome']) for r in reports]
                }
    except Exception as e:
        st.error(f"Error obteniendo tendencias financieras: {e}")
    return None

# Función para obtener datos de FRED
def get_macro_data(indicator, country, start_date, end_date):
    """Obtener datos macroeconómicos"""
    series_mapping = {
        ("GDP", "USA"): "GDP",
        ("Inflation", "USA"): "CPIAUCSL",
        ("Unemployment", "USA"): "UNRATE",
        ("Interest Rates", "USA"): "DFF"
    }
    
    try:
        series_id = series_mapping.get((indicator, country))
        if series_id:
            data = fred.get_series(series_id, start_date, end_date)
            return {
                'dates': data.index,
                'values': data.values
            }
    except Exception as e:
        st.error(f"Error obteniendo datos macro: {e}")
    return None

# ==================================================
# Funciones de Análisis
# ==================================================

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

# Función para calcular la asignación de riesgo jerárquico (HRP)
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
    
    # Ichimoku
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
    
    return df.fillna(method='ffill').fillna(method='bfill')

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
def calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate, period, interval):
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

# Función para calcular niveles dinámicos de stop loss y take profit
def calculate_dynamic_levels(data, symbol, confidence_level=0.95, risk_multiplier=3):
    returns = pd.Series(np.log(data[f'{symbol}_Close']).diff().dropna())
    conditional_vol = calculate_har_volatility(returns)
    z_score = norm.ppf(confidence_level)
    current_price = data[f'{symbol}_Close'].iloc[-1]
    
    stop_loss = current_price * np.exp(-z_score * conditional_vol)
    risk = current_price - stop_loss
    take_profit = current_price + (risk * risk_multiplier)
    
    return stop_loss, take_profit, conditional_vol

# ==================================================
# Funciones de Datos
# ==================================================

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

# ==================================================
# Interfaz de Usuario
# ==================================================

def display_trading_interface():
    st.set_page_config(page_title="Trading Platform Pro V5+", layout="wide")
    st.title("📈 Trading Platform Pro V5+")
    
    # Configuración global en sidebar
    with st.sidebar:
        st.warning("⚠️ PLATAFORMA EN DESARROLLO\nEsta es una versión demo - No usar para trading real.")
        symbols_input = st.text_input("Símbolos (separados por coma)", "AAPL,MSFT,GOOGL")
        symbols = [s.strip() for s in symbols_input.split(",")]
        state_period = st.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        state_interval = st.selectbox("Intervalo", ["1d", "5d", "1wk", "1mo"])
    
    # Obtener datos
    portfolio_data, info_dict = get_portfolio_data(symbols, state_period, state_interval)
    
    if portfolio_data is not None and not portfolio_data.empty:
        # Calcular métricas
        close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
        returns = portfolio_data[close_cols].pct_change().dropna()
        returns.columns = [col.replace('_Close', '') for col in returns.columns]
        weights = hierarchical_risk_parity(returns)
        
        # Crear tabs
        tabs = st.tabs(["Trading", "Análisis Técnico", "Noticias", "Fundamental", "Macro"])
        
        with tabs[0]:
            display_trading_tab(portfolio_data, info_dict, symbols, state_period, 
                             state_interval, weights)
        
        with tabs[1]:
            display_technical_analysis_tab(portfolio_data, symbols)
        
        with tabs[2]:
            display_news_tab()
        
        with tabs[3]:
            display_fundamental_tab(portfolio_data, symbols)
        
        with tabs[4]:
            display_macro_tab()

def display_technical_analysis_tab(portfolio_data, symbols):
    st.header("Análisis Técnico")
    selected_symbol = st.selectbox("Seleccionar Activo", symbols, key='technical_symbol')
    
    if selected_symbol:
        df = calculate_technical_indicators(portfolio_data, selected_symbol)
        if not df.empty:
            st.write(f"Indicadores Técnicos para {selected_symbol}")
            st.dataframe(df.tail())
            
            # Gráfico de RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df[f'{selected_symbol}_RSI'], name='RSI'))
            fig_rsi.update_layout(title=f"RSI para {selected_symbol}", xaxis_title="Fecha", yaxis_title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)

def display_news_tab():
    st.header("Noticias")
    category = st.selectbox("Seleccionar Categoría", ["financial", "macro", "political", "corporate", "commodities"])
    news = get_news_by_category(category)
    
    if news:
        for article in news:
            st.write(f"**{article['title']}**")
            st.write(f"*{article['publishedAt']}*")
            st.write(article['description'])
            st.write(f"[Leer más]({article['url']})")
            st.write("---")
    else:
        st.warning("No se encontraron noticias.")

def display_fundamental_tab(portfolio_data, symbols):
    st.header("Análisis Fundamental")
    selected_symbol = st.selectbox("Seleccionar Activo", symbols, key='fundamental_symbol')
    
    if selected_symbol:
        trends = get_financial_trends(selected_symbol)
        if trends:
            st.write(f"Tendencias Financieras para {selected_symbol}")
            st.write(f"Fechas: {trends['dates']}")
            st.write(f"Ingresos: {trends['revenues']}")
            st.write(f"Beneficio Neto: {trends['net_income']}")
        else:
            st.warning("No se encontraron datos fundamentales.")

def display_macro_tab():
    st.header("Datos Macro")
    indicator = st.selectbox("Seleccionar Indicador", ["GDP", "Inflation", "Unemployment", "Interest Rates"])
    start_date = st.date_input("Fecha de Inicio", datetime.now() - timedelta(days=365))
    end_date = st.date_input("Fecha de Fin", datetime.now())
    
    macro_data = get_macro_data(indicator, "USA", start_date, end_date)
    if macro_data:
        st.write(f"Datos de {indicator} para USA")
        st.write(f"Fechas: {macro_data['dates']}")
        st.write(f"Valores: {macro_data['values']}")
    else:
        st.warning("No se encontraron datos macro.")

def create_trading_chart(portfolio_data, symbol, chart_type, use_log, 
                        confidence_level, risk_multiplier):
    try:
        fig = go.Figure()
        
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(
                x=portfolio_data.index,
                open=portfolio_data[f'{symbol}_Open'],
                high=portfolio_data[f'{symbol}_High'],
                low=portfolio_data[f'{symbol}_Low'],
                close=portfolio_data[f'{symbol}_Close'],
                name=symbol
            ))
        elif chart_type == 'OHLC':
            fig.add_trace(go.Ohlc(
                x=portfolio_data.index,
                open=portfolio_data[f'{symbol}_Open'],
                high=portfolio_data[f'{symbol}_High'],
                low=portfolio_data[f'{symbol}_Low'],
                close=portfolio_data[f'{symbol}_Close'],
                name=symbol
            ))
        else:
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data[f'{symbol}_Close'],
                name=symbol
            ))

        # Calcular niveles
        stop_loss, take_profit, volatility = calculate_dynamic_levels(
            portfolio_data, symbol, confidence_level, risk_multiplier)

        # Añadir líneas de stop loss y take profit
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
            title=f"Trading View - {symbol}",
            xaxis_title="Fecha",
            yaxis_title="Precio",
            height=600,
            yaxis_type='log' if use_log else 'linear'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creando gráfico: {e}")
        return None

def display_trading_tab(portfolio_data, info_dict, symbols, period, interval, weights):
    st.header("Panel de Trading")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox("Tipo de Gráfico", ['Line', 'Candlestick', 'OHLC'])
        use_log = st.checkbox("Escala Logarítmica")
    with col2:
        confidence_level = st.slider("Nivel de Confianza (%)", 90, 99, 95) / 100
        risk_free_rate = st.number_input("Tasa Libre de Riesgo (%)", 0.0, 100.0, 2.0) / 100
    with col3:
        risk_multiplier = st.slider("Multiplicador de Riesgo", 2.0, 5.0, 3.0, 0.1)
    
    try:
        metrics = calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate, period, interval)
        display_metrics_panel(metrics)
    except Exception as e:
        st.error(f"Error en cálculo de métricas: {e}")
    
    selected_symbol = st.selectbox("Seleccionar Activo", symbols, key='trading_symbol')
    
    if chart := create_trading_chart(portfolio_data, selected_symbol, chart_type, 
                                   use_log, confidence_level, risk_multiplier):
        st.plotly_chart(chart, use_container_width=True)
        display_trading_info(portfolio_data, info_dict, selected_symbol)

def display_metrics_panel(metrics):
    with st.expander("📊 Panel de Métricas", expanded=True):
        if metrics:
            cols = ['Métrica', 'Portfolio', 'SPY', 'URTH']
            metrics_df = pd.DataFrame(columns=cols)
            metrics_df['Métrica'] = ['Sharpe', 'Sortino', 'Max Drawdown', 'VaR', 'CVaR']
            
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

def display_trading_info(portfolio_data, info_dict, symbol):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if info_dict.get(symbol):
            info = info_dict[symbol]
            st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industria:** {info.get('industry', 'N/A')}")
    
    with col2:
        current_price = portfolio_data[f'{symbol}_Close'].iloc[-1]
        st.metric("Precio Actual", f"${current_price:.2f}")
        
        quantity = st.number_input("Cantidad", min_value=1, value=1)
        total = current_price * quantity
        st.write(f"Total de la operación: ${total:,.2f}")
        
        if st.button("Ejecutar Orden"):
            st.success(f"Orden ejecutada: {quantity} {symbol} a ${current_price:.2f}")

if __name__ == "__main__":
    display_trading_interface()
