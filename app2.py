import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import ta

# Funciones de utilidad
def quasi_diag(link):
    link = link.astype(int)
    num_items = link[-1, 3]
    sort_ix = []
    sort_ix.extend([link[-1, 0], link[-1, 1]])
    
    for i in range(len(link) - 2, -1, -1):
        if link[i, 0] >= num_items:
            sort_ix.append(link[i, 1])
        elif link[i, 1] >= num_items:
            sort_ix.append(link[i, 0])
    
    return np.array([x for x in sort_ix if x < num_items])

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
                    portfolio_data[f"{ticker}_Close"] = df['Close']
                    portfolio_data[f"{ticker}_Volume"] = df['Volume']
                    info_dict[ticker] = stock.info
            except Exception as e:
                st.warning(f"Error al obtener datos para {ticker}: {e}")
                continue
        
        if portfolio_data.empty:
            st.error("No se pudieron obtener datos para ningún ticker")
            return None, None
            
        expected_columns = [f"{ticker}_Close" for ticker in tickers]
        if not all(col in portfolio_data.columns for col in expected_columns):
            missing = [ticker for ticker in tickers 
                      if f"{ticker}_Close" not in portfolio_data.columns]
            st.warning(f"Faltantes datos para: {', '.join(missing)}")
        
        return portfolio_data, info_dict
    except Exception as e:
        st.error(f"Error al obtener datos del portafolio: {e}")
        return None, None

# Funciones de optimización de cartera
def hierarchical_risk_parity(returns):
    try:
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(method='ffill').fillna(method='bfill')
        
        if returns.empty or returns.isna().all().all():
            raise ValueError("No hay datos suficientes para el cálculo")
        
        corr = returns.corr()
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
        dist = np.nan_to_num(dist, nan=0.0)
        
        link = linkage(squareform(dist), method='ward')
        sort_ix = quasi_diag(link)
        
        var = returns.var()
        var = var.replace(0, np.finfo(float).eps)
        weights = 1/var
        weights = weights/weights.sum()
        
        weights_series = pd.Series(weights.values, index=returns.columns)
        
        if not weights_series.isna().any():
            return weights_series
        else:
            raise ValueError("Error en el cálculo de pesos")
            
    except Exception as e:
        st.error(f"Error en HRP: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

def hierarchical_risk_clustering(returns):
    try:
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        link = linkage(squareform(dist), 'ward')
        
        n_assets = len(returns.columns)
        n_clusters = min(int(np.sqrt(n_assets)), n_assets)
        
        # Calcular varianzas por cluster
        cluster_var = returns.var()
        inv_var = 1 / cluster_var
        weights = inv_var / inv_var.sum()
        
        return pd.Series(weights, index=returns.columns)
    except Exception as e:
        st.error(f"Error en HRC: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

def equal_weighted(returns):
    n_assets = len(returns.columns)
    weights = np.ones(n_assets) / n_assets
    return pd.Series(weights, index=returns.columns)

def mean_variance(returns, risk_free_rate=0.03):
    try:
        n_assets = len(returns.columns)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        def portfolio_stats(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            return -sharpe_ratio
        
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_stats, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=returns.columns)
        else:
            raise ValueError("Optimización no convergió")
    except Exception as e:
        st.error(f"Error en Mean-Variance: {e}")
        return pd.Series({col: 1.0/len(returns.columns) for col in returns.columns})

def calculate_portfolio_metrics(returns, weights):
    try:
        portfolio_return = (returns * weights).sum(axis=1)
        
        annual_return = portfolio_return.mean() * 252
        annual_vol = portfolio_return.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        cum_returns = (1 + portfolio_return).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'Rentabilidad Anual': annual_return,
            'Volatilidad Anual': annual_vol,
            'Ratio Sharpe': sharpe,
            'Máximo Drawdown': max_drawdown,
            'Rentabilidad Total': cum_returns.iloc[-1] - 1
        }
        return metrics
    except Exception as e:
        st.error(f"Error en cálculo de métricas: {e}")
        return {
            'Rentabilidad Anual': 0,
            'Volatilidad Anual': 0,
            'Ratio Sharpe': 0,
            'Máximo Drawdown': 0,
            'Rentabilidad Total': 0
        }

def calculate_technical_indicators(df, symbol):
    try:
        df = df.copy()
        df = df.sort_index()
        
        close_price = df[f'{symbol}_Close']
        volume = df[f'{symbol}_Volume']
        
        try:
            df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(
                high=close_price, 
                low=close_price, 
                close=close_price, 
                volume=volume,
                window=14
            )
        except:
            df[f'{symbol}_VWAP'] = close_price.rolling(window=14).mean()
        
        df[f'{symbol}_EMA20'] = ta.trend.ema_indicator(close_price, window=20)
        df[f'{symbol}_EMA50'] = ta.trend.ema_indicator(close_price, window=50)
        df[f'{symbol}_SMA20'] = close_price.rolling(window=20).mean()
        df[f'{symbol}_SMA50'] = close_price.rolling(window=50).mean()
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error en cálculo de indicadores técnicos: {e}")
        return df

# Configuración de la página
st.set_page_config(page_title="Plataforma de Trading Avanzada", layout="wide")
st.title("📈 Plataforma de Trading Avanzada")

# Configuración de la barra lateral
with st.sidebar:
    st.header("Configuración")
    symbols_input = st.text_input("Símbolos (separados por coma)", value="AAPL,MSFT,GOOGL,AMZN")
    symbols = [s.strip() for s in symbols_input.split(",")]
    
    period = st.selectbox(
        "Período",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    interval = st.selectbox(
        "Intervalo",
        options=["1d", "5d", "1wk", "1mo"],
        index=0
    )

# Obtener datos
portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    
    # Obtener datos de SPY y URTH
    benchmark_data = get_portfolio_data(['SPY', 'URTH'], period, interval)[0]
    if benchmark_data is not None:
        benchmark_returns = benchmark_data.filter(like='Close').pct_change().dropna()
        benchmark_returns.columns = [col.replace('_Close', '') for col in benchmark_returns.columns]
    
    # Calcular pesos para cada modelo
    weights_dict = {
        'HRP': hierarchical_risk_parity(returns),
        'HRC': hierarchical_risk_clustering(returns),
        'Equal Weight': equal_weighted(returns),
        'Mean-Variance': mean_variance(returns)
    }
    
    # Calcular métricas para cada cartera
    metrics_dict = {
        method: calculate_portfolio_metrics(returns, weights)
        for method, weights in weights_dict.items()
    }
    
    # Crear pestañas para la interfaz
    tab1, tab2, tab3 = st.tabs(["Análisis de Cartera", "Métricas de Rendimiento", "Análisis Técnico"])
    
    with tab1:
        st.subheader("Composición de Carteras")
        selected_method = st.selectbox("Seleccionar Método de Optimización", list(weights_dict.keys()))
        
        # Mostrar pesos de la cartera seleccionada
        weights_df = pd.DataFrame({
            'Activo': weights_dict[selected_method].index,
            'Peso (%)': (weights_dict[selected_method].values * 100).round(2)
        })
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=weights_df['Activo'],
                values=weights_df['Peso (%)'],
                textinfo='label+percent'
            )])
            fig.update_layout(title=f"Distribución de la Cartera - {selected_method}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(weights_df)
    
    with tab2:
        st.subheader("Comparación de Carteras")
        
        # Mostrar métricas
        metrics_df = pd.DataFrame(metrics_dict).round(4)
        st.dataframe(metrics_df)
        
        # Gráfico de rendimientos acumulados
        fig = go.Figure()
        
        # Añadir rendimientos de las carteras
        for method, weights in weights_dict.items():
            portfolio_returns = (returns * weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                name=method
            ))
        
        # Añadir benchmarks
        if benchmark_data is not None:
            for col in benchmark_returns.columns:
                cumulative_returns = (1 + benchmark_returns[col]).cumprod()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    name=col,
                    line=dict(dash='dash')
                ))
        
        fig.update_layout(
            title="Rendimientos Acumulados por Método",
            xaxis_title="Fecha",
            yaxis_title="Retorno Acumulado (log)",
            yaxis_type="log",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        selected_symbol = st.selectbox("Seleccionar Activo", symbols)
        
        technical_data = calculate_technical_indicators(portfolio_data, selected_symbol)
        
        fig = go.Figure()
        
        # Precio y indicadores técnicos
        fig.add_trace(go.Scatter(
            x=technical_data.index,
            y=technical_data[f'{selected_symbol}_Close'],
            name='Precio'
        ))
        
        for indicator in ['VWAP', 'EMA20', 'EMA50', 'SMA20', 'SMA50']:
            fig.add_trace(go.Scatter(
                x=technical_data.index,
                y=technical_data[f'{selected_symbol}_{indicator}'],
                name=indicator
            ))
        
        fig.update_layout(
            title=f"Análisis Técnico - {selected_symbol}",
            xaxis_title="Fecha",
            yaxis_title="Precio",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Panel de operaciones
        st.subheader("Panel de Operaciones")
        col1, col2 = st.columns(2)
        
        with col1:
            operation = st.radio("Tipo de Operación", ["Comprar", "Vender"])
            quantity = st.number_input("Cantidad", min_value=1, value=1)
        
        with col2:
            price = st.number_input(
                "Precio",
                min_value=0.01,
                value=float(technical_data[f'{selected_symbol}_Close'].iloc[-1]),
                format="%.2f"
            )
            
            total = price * quantity
            st.write(f"Total de la operación: ${total:,.2f}")
        
        if st.button("Ejecutar Orden"):
            st.success(f"Orden de {operation} ejecutada: {quantity} {selected_symbol} a ${price:.2f}")
            st.write(f"Total: ${total:,.2f}")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.write("Desarrollado con ❤️ usando Streamlit")
st.sidebar.markdown("---")
st.sidebar.info("Esta es una plataforma de demostración. No utilizar para trading real.")
