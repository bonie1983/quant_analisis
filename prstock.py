import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # División de datos
from sklearn.linear_model import LogisticRegression # Modelo de Regresión Logística
from sklearn.metrics import accuracy_score # Métrica de Precisión
import ta  # Librería de Análisis Técnico

# ============================================================
# 1. CONFIGURACIÓN INICIAL
# ============================================================
API_KEY = "XMOE4LFC24PF7LJF" #
SYMBOL = "MSFT"
MARKET = "USD"
FUNCTION = "TIME_SERIES_DAILY"

MARKET_DAYS = 252  
TASA_LIBRE_RIESGO = 0.04  

URL = (
    f"https://www.alphavantage.co/query?"
    f"function={FUNCTION}&"
    f"symbol={SYMBOL}&"
    f"market={MARKET}&"
    f"apikey={API_KEY}"
)

# ============================================================
# 2. DESCARGA DE DATOS DESDE ALPHA VANTAGE
# ============================================================
try:
    print(f"Intentando conectar a Alpha Vantage para obtener datos de {SYMBOL}...")
    response = requests.get(URL)
    response.raise_for_status()
    data = response.json()
    print("Conexión exitosa. Datos recibidos.")
except requests.exceptions.RequestException as e:
    print(f"Error al conectar con la API: {e}")
    exit()

# ============================================================
# 3. LIMPIEZA Y PREPARACIÓN DEL DATAFRAME
# ============================================================
try:
    key_prices = 'Time Series (Daily)'
    df = pd.DataFrame(data[key_prices]).T
    df = df.iloc[:, :5]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['daily_returns'] = df['close'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['Median_10'] = df['close'].rolling(10).median()

    df['Target'] = (df['daily_returns'].shift(-1) > 0).astype(int)
except KeyError:
    print("Error: La API no devolvió los datos esperados.")
    exit()

# ============================================================
# 4. CÁLCULO DE INDICADORES TÉCNICOS (usando ta)
# ============================================================
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['stoch'] = ta.momentum.StochasticOscillator(
    high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3
).stoch()
df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()

df['macd'] = ta.trend.MACD(df['close']).macd()
df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()

df['bb_high'] = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2).bollinger_hband()
df['bb_low'] = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2).bollinger_lband()
df['atr'] = ta.volatility.AverageTrueRange(
    high=df['high'], low=df['low'], close=df['close'], window=14
).average_true_range()

df['mfi'] = ta.volume.MFIIndicator(
    high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14
).money_flow_index()
df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

df = df.dropna()

print("\n--- Vista Preliminar (Datos listos para ML) ---")
print(df.tail())

# ============================================================
# 5. ANÁLISIS DE RIESGO
# ============================================================
volatilidad_diaria = df['daily_returns'].std()
volatilidad_anualizada = volatilidad_diaria * np.sqrt(MARKET_DAYS)
rendimiento_anualizado = df['daily_returns'].mean() * MARKET_DAYS
ratio_sharpe = (rendimiento_anualizado - TASA_LIBRE_RIESGO) / volatilidad_anualizada

rolling_max = df['cumulative_returns'].cummax()
drawdown = (df['cumulative_returns'] / rolling_max) - 1
mdd = drawdown.min()
ratio_calmar = rendimiento_anualizado / abs(mdd)

print("\n--- INFORME FINANCIERO ---")
print(f"Volatilidad Anualizada: {volatilidad_anualizada:.2%}")
print(f"Rendimiento Anualizado: {rendimiento_anualizado:.2%}")
print(f"Ratio Sharpe: {ratio_sharpe:.4f}")
print(f"Máxima Caída (MDD): {mdd:.2%}")
print(f"Ratio Calmar: {ratio_calmar:.4f}")

# ============================================================
# 6. MACHINE LEARNING
# ============================================================
FEATURES = [
    'close', 'volume', 'SMA_20', 'SMA_50', 'Median_10',
    'rsi', 'stoch', 'roc', 'macd', 'macd_signal', 'ema_12', 'ema_26',
    'bb_high', 'bb_low', 'atr', 'mfi', 'obv'
]

X = df[FEATURES]
Y = df['Target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
modelo = LogisticRegression(solver='lbfgs', max_iter=1000)
modelo.fit(X_train, Y_train)

Y_pred = modelo.predict(X_test)
precision = accuracy_score(Y_test, Y_pred)

X_today = df[FEATURES].iloc[-1].values.reshape(1, -1)
prediccion_manana = modelo.predict(X_today)[0]

print("\n--- RESULTADOS DEL MODELO ---")
print(f"Precisión: {precision:.2%}")
if prediccion_manana == 1:
    print("📈 El modelo predice que el precio de mañana SUBIRÁ.")
else:
    print("📉 El modelo predice que el precio de mañana BAJARÁ o se MANTENDRÁ.")

# ============================================================
# 7. VISUALIZACIÓN
# ============================================================
prediccion_texto = "SUBIRÁ 🟢" if prediccion_manana == 1 else "BAJARÁ 🔴"
dashboard_text = (
    f"--- Dashboard ({SYMBOL}) ---\n"
    f"Volatilidad: {volatilidad_anualizada:.2%}\n"
    f"Rendimiento: {rendimiento_anualizado:.2%}\n"
    f"Sharpe: {ratio_sharpe:.2f}\n"
    f"Calmar: {ratio_calmar:.2f}\n"
    f"MDD: {mdd:.2%}\n"
    f"Precisión ML: {precision:.2%}\n"
    f"Predicción: {prediccion_texto}"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(df.index, df['close'], color='blue', label='Precio de Cierre')
ax1.plot(df.index, df['SMA_20'], color='orange', linestyle='--', label='SMA 20')
ax1.plot(df.index, df['SMA_50'], color='red', linestyle='--', label='SMA 50')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.bar(df.index, df['volume'], color='gray', alpha=0.6)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.figtext(0.70, 0.90, dashboard_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7, ec="gray"))
plt.tight_layout(rect=[0, 0, 0.80, 1])
plt.show()

print("\n✅ Script ejecutado correctamente con Dashboard y Predicción.")
