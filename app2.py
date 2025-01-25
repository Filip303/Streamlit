import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import ta

def hierarchical_risk_parity(returns):
    """
    Implementa HRP con manejo robusto de datos
    """
    # Limpieza de datos
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(method='ffill').fillna(method='bfill')
    
    # Asegurarse que no hay valores no finitos
    if not np.isfinite(returns).all().all():
        raise ValueError("Hay valores no finitos después de la limpieza")
    
    # Matriz de correlación con manejo de errores
    corr = returns.corr()
    
    # Verificar si la matriz de correlación es válida
    if not np.isfinite(corr).all().all():
        corr = returns.fillna(0).corr()
    
    # Convertir correlación a distancia con verificación
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1)).values
    
    # Clustering jerárquico
    link = linkage(squareform(dist, checks=False), 'ward')
    
    # Ordenamiento y cálculo de pesos
    sort_ix = quasi_diag(link)
    
    # Cálculo de pesos más robusto
    var = returns.var()
    weights = 1/np.clip(var, 1e-6, np.inf)  # Evitar división por cero
    weights = weights/weights.sum()
    
    return pd.Series(weights.values, index=returns.columns)

def quasi_diag(link):
    link = link.astype(int)
    num_items = link[-1, 3]
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    
    for i in range(len(link) - 2, -1, -1):
        if link[i, 0] >= num_items:
            sort_ix.append(link[i, 1])
        elif link[i, 1] >= num_items:
            sort_ix.append(link[i, 0])
    
    return sort_ix.map(lambda x: x if x < num_items else None).dropna()

@st.cache_data
def get_portfolio_data(tickers, period, interval):
    try:
        portfolio_data = pd.DataFrame()
        info_dict = {}
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            if not df.empty:
                portfolio_data[f"{ticker}_Close"] = df['Close']
                portfolio_data[f"{ticker}_Volume"] = df['Volume']
                info_dict[ticker] = stock.info
        
        if portfolio_data.empty:
            raise Exception("No se pudieron obtener datos para ningún ticker")
            
        return portfolio_data, info_dict
    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return None, None

def calculate_metrics(portfolio_data, weights):
    """Calcula métricas con manejo robusto de errores"""
    try:
        returns = portfolio_data.filter(like='Close').pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        weights_aligned = pd.Series(weights, index=returns.columns)
        portfolio_return = returns.dot(weights_aligned)
        
        risk_free_rate = 0.02
        
        # Cálculos con manejo de errores
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
        
        return {
            'Sharpe': np.clip(sharpe, -100, 100),
            'Sortino': np.clip(sortino, -100, 100),
            'Calmar': np.clip(calmar, -100, 100),
            'Max Drawdown': max_drawdown
        }
    except Exception as e:
        st.error(f"Error en cálculo de métricas: {e}")
        return {
            'Sharpe': 0,
            'Sortino': 0,
            'Calmar': 0,
            'Max Drawdown': 0
        }

def calculate_technical_indicators(df, symbol):
    """Calcula indicadores técnicos con manejo de errores"""
    try:
        df = df.copy()
        close_price = df[f'{symbol}_Close']
        volume = df[f'{symbol}_Volume']
        
        df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(
            high=close_price, 
            low=close_price, 
            close=close_price, 
            volume=volume
        )
        
        df[f'{symbol}_EMA20'] = ta.trend.ema_indicator(close_price, window=20)
        df[f'{symbol}_EMA50'] = ta.trend.ema_indicator(close_price, window=50)
        df[f'{symbol}_SMA20'] = ta.trend.sma_indicator(close_price, window=20)
        df[f'{symbol}_SMA50'] = ta.trend.sma_indicator(close_price, window=50)
        
        # Limpiar NaN y valores infinitos
        for col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        
        return df
    except Exception as e:
        st.error(f"Error en cálculo de indicadores técnicos: {e}")
        return df

# Configuración de la página
st.set_page_config(page_title="Plataforma de Trading Avanzada", layout="wide")
st.title("📈 Plataforma de Trading Avanzada")

# Sidebar para configuración
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
    # Preparar datos para HRP
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    
    # Calcular pesos usando HRP con manejo de errores
    try:
        weights = hierarchical_risk_parity(returns)
    except Exception as e:
        st.error(f"Error en el cálculo de HRP: {e}")
        weights = pd.Series({symbol: 1/len(symbols) for symbol in symbols})
    
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Análisis de Cartera", "Métricas de Rendimiento", "Análisis Técnico"])
    
    with tab1:
        st.subheader("Composición Optimizada de la Cartera (HRP)")
        weights_df = pd.DataFrame({'Activo': weights.index, 'Peso': weights.values * 100})
        
        fig = go.Figure(data=[go.Pie(labels=weights_df['Activo'],
                                   values=weights_df['Peso'],
                                   textinfo='label+percent')])
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(weights_df.round(2))
    
    with tab2:
        metrics = calculate_metrics(portfolio_data, weights)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ratio Sharpe", f"{metrics['Sharpe']:.2f}")
        with col2:
            st.metric("Ratio Sortino", f"{metrics['Sortino']:.2f}")
        with col3:
            st.metric("Ratio Calmar", f"{metrics['Calmar']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        
        # Cálculo de retornos de la cartera
        close_prices = portfolio_data.filter(like='Close')
        close_prices.columns = [col.replace('_Close', '') for col in close_prices.columns]
        portfolio_returns = (close_prices.pct_change() * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index,
                               y=cumulative_returns,
                               name='Retorno Acumulado'))
        fig.update_layout(title="Evolución de la Cartera",
                         xaxis_title="Fecha",
                         yaxis_title="Retorno Acumulado")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        selected_symbol = st.selectbox("Seleccionar Activo", symbols)
        
        technical_data = calculate_technical_indicators(portfolio_data.copy(), selected_symbol)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=technical_data.index,
                               y=technical_data[f'{selected_symbol}_Close'],
                               name='Precio'))
        
        fig.add_trace(go.Scatter(x=technical_data.index,
                               y=technical_data[f'{selected_symbol}_VWAP'],
                               name='VWAP'))
        
        fig.add_trace(go.Scatter(x=technical_data.index,
                               y=technical_data[f'{selected_symbol}_EMA20'],
                               name='EMA 20'))
        fig.add_trace(go.Scatter(x=technical_data.index,
                               y=technical_data[f'{selected_symbol}_EMA50'],
                               name='EMA 50'))
        
        fig.add_trace(go.Scatter(x=technical_data.index,
                               y=technical_data[f'{selected_symbol}_SMA20'],
                               name='SMA 20'))
        fig.add_trace(go.Scatter(x=technical_data.index,
                               y=technical_data[f'{selected_symbol}_SMA50'],
                               name='SMA 50'))
        
        fig.update_layout(title=f"Análisis Técnico - {selected_symbol}",
                         xaxis_title="Fecha",
                         yaxis_title="Precio",
                         height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        df_display = pd.DataFrame({
            'Precio': technical_data[f'{selected_symbol}_Close'],
            'VWAP': technical_data[f'{selected_symbol}_VWAP'],
            'EMA20': technical_data[f'{selected_symbol}_EMA20'],
            'EMA50': technical_data[f'{selected_symbol}_EMA50'],
            'SMA20': technical_data[f'{selected_symbol}_SMA20'],
            'SMA50': technical_data[f'{selected_symbol}_SMA50']
        }).round(2)
        
        st.dataframe(df_display)
        
        # Señales de trading con manejo de NaN
        df_display['Señal'] = 'Mantener'
        df_display.loc[technical_data[f'{selected_symbol}_EMA20'].fillna(0) > 
                      technical_data[f'{selected_symbol}_EMA50'].fillna(0), 'Señal'] = 'Comprar'
        df_display.loc[technical_data[f'{selected_symbol}_EMA20'].fillna(0) < 
                      technical_data[f'{selected_symbol}_EMA50'].fillna(0), 'Señal'] = 'Vender'
        
        st.subheader("Señales de Trading")
        latest_signal = df_display['Señal'].iloc[-1]
        st.info(f"Señal actual para {selected_symbol}: {latest_signal}")
        
        st.subheader("Panel de Operaciones")
        col1, col2 = st.columns(2)
        
        with col1:
            operation = st.radio("Tipo de Operación", ["Comprar", "Vender"])
            quantity = st.number_input("Cantidad", min_value=1, value=1)
        
        with col2:
            price = st.number_input("Precio",
                                  min_value=0.01,
                                  value=float(technical_data[f'{selected_symbol}_Close'].iloc[-1]),
                                  format="%.2f")
            
            total = price * quantity
            st.write(f"Total de la operación: ${total:,.2f}")
        
        if st.button("Ejecutar Orden"):
            st.success(f"Orden de {operation} ejecutada: {quantity} {selected_symbol} a ${price:.2f}")
            st.write(f"Total: ${total:,.2f}")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.write("Desarrollado con ❤️ usando Streamlit")
    st.sidebar.markdown("---")
    st.sidebar.info("Esta es una plataforma de demostración. No utilizar para trading real.")
