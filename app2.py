import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pypfopt import HierarchicalRiskParity
from pypfopt import risk_metrics
import ta

# Configuración de la página
st.set_page_config(page_title="Plataforma de Trading Avanzada", layout="wide")
st.title("📈 Plataforma de Trading Avanzada")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    symbols_input = st.text_input("Símbolos (separados por coma)", value="AAPL,MSFT,GOOGL")
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

@st.cache_data
def get_portfolio_data(tickers, period, interval):
    try:
        portfolio_data = pd.DataFrame()
        info_dict = {}
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            portfolio_data[f"{ticker}_Close"] = df['Close']
            portfolio_data[f"{ticker}_Volume"] = df['Volume']
            info_dict[ticker] = stock.info
            
        return portfolio_data, info_dict
    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return None, None

def calculate_metrics(portfolio_data, weights):
    returns = portfolio_data.filter(like='Close').pct_change()
    portfolio_return = returns.dot(weights)
    
    risk_free_rate = 0.02
    sharpe = np.sqrt(252) * (portfolio_return.mean() - risk_free_rate/252) / portfolio_return.std()
    
    downside_returns = portfolio_return[portfolio_return < 0]
    sortino = np.sqrt(252) * (portfolio_return.mean() - risk_free_rate/252) / downside_returns.std()
    
    cum_returns = (1 + portfolio_return).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    calmar = -252 * portfolio_return.mean() / max_drawdown
    
    return {
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def calculate_technical_indicators(df, symbol):
    df[f'{symbol}_VWAP'] = ta.volume.volume_weighted_average_price(
        high=df[f'{symbol}_Close'],
        low=df[f'{symbol}_Close'],
        close=df[f'{symbol}_Close'],
        volume=df[f'{symbol}_Volume']
    )
    
    df[f'{symbol}_EMA20'] = ta.trend.ema_indicator(df[f'{symbol}_Close'], window=20)
    df[f'{symbol}_EMA50'] = ta.trend.ema_indicator(df[f'{symbol}_Close'], window=50)
    df[f'{symbol}_SMA20'] = ta.trend.sma_indicator(df[f'{symbol}_Close'], window=20)
    df[f'{symbol}_SMA50'] = ta.trend.sma_indicator(df[f'{symbol}_Close'], window=50)
    
    return df

# Obtener datos
portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    # Optimización HRP
    returns = portfolio_data.filter(like='Close').pct_change().dropna()
    hrp = HierarchicalRiskParity()
    hrp.fit(returns)
    weights = hrp.optimize()
    
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Análisis de Cartera", "Métricas de Rendimiento", "Análisis Técnico"])
    
    with tab1:
        st.subheader("Composición Optimizada de la Cartera")
        weights_df = pd.DataFrame(list(weights.items()), columns=['Activo', 'Peso'])
        weights_df['Peso'] = weights_df['Peso'].round(4) * 100
        
        fig = go.Figure(data=[go.Pie(labels=weights_df['Activo'],
                                   values=weights_df['Peso'],
                                   textinfo='label+percent')])
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(weights_df)
    
    with tab2:
        metrics = calculate_metrics(portfolio_data, list(weights.values()))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Ratio Sharpe", f"{metrics['Sharpe']:.2f}")
        with col2:
            st.metric("Ratio Sortino", f"{metrics['Sortino']:.2f}")
        with col3:
            st.metric("Ratio Calmar", f"{metrics['Calmar']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        
        portfolio_returns = portfolio_data.filter(like='Close').pct_change().dot(list(weights.values()))
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
        
        df_display['Señal'] = 'Mantener'
        df_display.loc[technical_data[f'{selected_symbol}_EMA20'] > technical_data[f'{selected_symbol}_EMA50'], 'Señal'] = 'Comprar'
        df_display.loc[technical_data[f'{selected_symbol}_EMA20'] < technical_data[f'{selected_symbol}_EMA50'], 'Señal'] = 'Vender'
        
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
