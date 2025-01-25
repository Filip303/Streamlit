import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import ta

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

def calculate_metrics(portfolio_data, weights):
    try:
        returns = portfolio_data.filter(like='Close').pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        weights_dict = {col.replace('_Close', ''): weight 
                       for col, weight in zip(returns.columns, weights)}
        portfolio_return = sum(returns[f"{symbol}_Close"] * weights_dict[symbol] 
                             for symbol in weights_dict.keys())
        
        risk_free_rate = 0.02
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
        
        metrics = {
            'Sharpe': np.clip(sharpe, -100, 100),
            'Sortino': np.clip(sortino, -100, 100),
            'Calmar': np.clip(calmar, -100, 100),
            'Max Drawdown': max_drawdown
        }
        
        metrics = {k: 0 if np.isnan(v) else v for k, v in metrics.items()}
        return metrics
        
    except Exception as e:
        st.error(f"Error en cálculo de métricas: {e}")
        return {
            'Sharpe': 0,
            'Sortino': 0,
            'Calmar': 0,
            'Max Drawdown': 0
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

portfolio_data, info_dict = get_portfolio_data(symbols, period, interval)

if portfolio_data is not None and not portfolio_data.empty:
    close_cols = [col for col in portfolio_data.columns if col.endswith('_Close')]
    returns = portfolio_data[close_cols].pct_change().dropna()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    
    try:
        weights = hierarchical_risk_parity(returns)
    except Exception as e:
        st.error(f"Error en el cálculo de HRP: {e}")
        weights = pd.Series({symbol: 1/len(symbols) for symbol in symbols})
    
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
        
        close_prices = portfolio_data.filter(like='Close')
        close_prices.columns = [col.replace('_Close', '') for col in close_prices.columns]
        portfolio_returns = (close_prices.pct_change() * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Obtener datos del SPY y URTH
        spy_data = yf.download("SPY", period=period, interval=interval)['Close']
        urth_data = yf.download("URTH", period=period, interval=interval)['Close']
        
        # Calcular retornos acumulados para SPY y URTH
        spy_cum_returns = (1 + spy_data.pct_change().fillna(0)).cumprod()
        urth_cum_returns = (1 + urth_data.pct_change().fillna(0)).cumprod()
        
        # Gráfico de retornos acumulados en escala logarítmica
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative_returns.index,
                               y=cumulative_returns,
                               name='Cartera'))
        fig.add_trace(go.Scatter(x=spy_cum_returns.index,
                               y=spy_cum_returns,
                               name='SPY'))
        fig.add_trace(go.Scatter(x=urth_cum_returns.index,
                               y=urth_cum_returns,
                               name='URTH'))
        fig.update_layout(title="Comparación de Retornos Acumulados (Escala Logarítmica)",
                         xaxis_title="Fecha",
                         yaxis_title="Retorno Acumulado",
                         yaxis_type="log",
                         height=600)
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
        
        fig.update_layout(title=f"Análisis Técnico - {selected_symbol} (Escala Logarítmica)",
                         xaxis_title="Fecha",
                         yaxis_title="Precio",
                         yaxis_type="log",
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
        
        # Señales de trading
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
