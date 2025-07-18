import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Fungsi Load & Preprocess
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Gold Price (2013-2023).csv')
    df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = df['Price'].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

# -------------------------
# Mulai Aplikasi
# -------------------------
st.title("📈 Prediksi Harga Emas dengan LSTM")
st.markdown("Prediksi harga emas berdasarkan data historis 2013–2023.")

# Load model dan data
model = load_model('model_lstm_emas.h5')
df = load_data()

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Price']])
window_size = 60

# Ambil 60 hari terakhir untuk input prediksi
last_sequence = scaled_data[-window_size:]
current_input = last_sequence.reshape(1, window_size, 1)

# Prediksi 12 bulan ke depan
# Pilih jumlah bulan ke depan yang ingin diprediksi
n_months = st.slider("Berapa bulan ke depan ingin diprediksi?", min_value=1, max_value=36, value=12)
future_steps = n_months  # Jika 1 langkah = 1 bulan

predicted_prices = []

for _ in range(future_steps):
    next_pred = model.predict(current_input, verbose=0)[0][0]
    predicted_prices.append(next_pred)
    current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)

# Invers transform hasil prediksi
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

# Buat tanggal prediksi
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': predicted_prices
})

# -------------------------
# Plot Hasil
# -------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Price'],
    mode='lines',
    name='Harga Historis',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=forecast_df['Date'],
    y=forecast_df['Forecast'],
    mode='lines+markers',
    name='Prediksi Harga Emas',
    line=dict(color='orange', dash='dash')
))

fig.update_layout(
    title='📊 Grafik Pediksi Harga Emas ',
    xaxis_title='Tanggal',
    yaxis_title='Harga Emas (USD/oz)',
    template='plotly_dark',
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(forecast_df)
