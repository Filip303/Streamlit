# requirements.txt
# streamlit
# pandas
# yfinance
# plotly

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Plataforma de Trading", layout="wide")

# Título y descripción
st.title("📈 Plataforma de Trading")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    symbol = st.text_input("Símbolo", value="AAPL")
    period = st.selectbox(
        "Período",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    )
    interval = st.selectbox(
        "Intervalo",
        options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    )

# Función para obtener datos
@st.cache_data
def get_stock_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df, stock.info
    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return None, None

# Obtener datos
df, info = get_stock_data(symbol, period, interval)

if df is not None and not df.empty:
    # Crear dos columnas
    col1, col2 = st.columns([2, 1])

    with col1:
        # Gráfico de velas
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'])])

        fig.update_layout(title=f"Gráfico de Velas - {symbol}",
                         xaxis_title="Fecha",
                         yaxis_title="Precio",
                         height=600)

        st.plotly_chart(fig, use_container_width=True)

        # Volumen
        volume_fig = go.Figure(data=[go.Bar(x=df.index, y=df['Volume'])])
        volume_fig.update_layout(title="Volumen",
                               xaxis_title="Fecha",
                               yaxis_title="Volumen",
                               height=200)

        st.plotly_chart(volume_fig, use_container_width=True)

    with col2:
        # Información del activo
        if info:
            st.subheader("Información del Activo")
            st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industria:** {info.get('industry', 'N/A')}")
            st.write(f"**Capitalización:** ${info.get('marketCap', 0):,.2f}")

        # Métricas clave
        st.subheader("Métricas Clave")
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Último Precio", f"${df['Close'].iloc[-1]:.2f}",
                     f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):.2f}%")
            st.metric("Volumen", f"{df['Volume'].iloc[-1]:,.0f}")

        with col_b:
            st.metric("Apertura", f"${df['Open'].iloc[-1]:.2f}")
            st.metric("Máximo", f"${df['High'].iloc[-1]:.2f}")

        # Panel de operaciones
        st.subheader("Panel de Operaciones")
        operation = st.radio("Tipo de Operación", ["Comprar", "Vender"])
        quantity = st.number_input("Cantidad", min_value=1, value=1)
        price = st.number_input("Precio", min_value=0.01, value=float(df['Close'].iloc[-1]))

        if st.button("Ejecutar Orden"):
            st.success(f"Orden de {operation} ejecutada: {quantity} {symbol} a ${price:.2f}")