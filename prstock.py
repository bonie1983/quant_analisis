import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. CONFIGURACIÓN: REEMPLAZA CON TU CLAVE Y DEFINE EL ACTIVO
API_KEY = "XMOE4LFC24PF7LJF" 
SYMBOL = "MSFT" # Ejemplo de Símbolo de Acción
MARKET = "USD"
FUNCTION = "TIME_SERIES_DAILY" # Usar "DIGITAL_CURRENCY_DAILY" para BTC/ETH

# Definiciones para análisis de riesgo
MARKET_DAYS = 252  # Días hábiles en un año para acciones
TASA_LIBRE_RIESGO = 0.04  # 4% anual

# 2. CONSTRUCCIÓN DE LA URL DE LA PETICIÓN
URL = (
    f"https://www.alphavantage.co/query?"
    f"function={FUNCTION}&"
    f"symbol={SYMBOL}&"
    f"market={MARKET}&"
    f"apikey={API_KEY}"
)

# 3. REALIZAR LA PETICIÓN Y OBTENER JSON
data = {} 
try:
    print(f"Intentando conectar a Alpha Vantage para obtener datos de {SYMBOL}...")
    response = requests.get(URL)
    response.raise_for_status() 

    data = response.json()
    print("Conexión exitosa. Datos recibidos.")
    
except requests.exceptions.RequestException as e:
    print(f"Error al conectar con la API: {e}")
    exit() 

# 4. CONVERSIÓN, LIMPIEZA, ANÁLISIS Y PREPARACIÓN DE ML
try:
    # --- LÓGICA CONDICIONAL PARA DETECTAR LA CLAVE CORRECTA ---
    if FUNCTION == "DIGITAL_CURRENCY_DAILY":
        key_prices = 'Time Series (Digital Currency Daily)'
        COLUMNAS_FINALES = ['open', 'high', 'low', 'close', 'volume'] 
    elif FUNCTION == "TIME_SERIES_DAILY":
        key_prices = 'Time Series (Daily)'
        COLUMNAS_FINALES = ['open', 'high', 'low', 'close', 'volume'] 
    else:
        raise ValueError("Función de API no reconocida.")
    
    # JSON a DataFrame y transposición (.T)
    df = pd.DataFrame(data[key_prices]).T
    df = df.iloc[:, :len(COLUMNAS_FINALES)]
    df.columns = COLUMNAS_FINALES
    df.index = pd.to_datetime(df.index) 
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    df = df.sort_index(ascending=True)
    
    # --- FASE 2: CÁLCULO DE RENDIMIENTO Y RIESGO (Retornos) ---
    df['daily_returns'] = df['close'].pct_change()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # --- FASE 3: CREACIÓN DE FEATURES Y TARGET ---
    
    # CALCULO DE MEDIAS MÓVILES Y MEDIANA MÓVIL (Features)
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['Median_10'] = df['close'].rolling(10).median()

    # Creación del Target (¿Subirá el precio mañana?)
    df['Target'] = df['daily_returns'].shift(-1) > 0 
    df['Target'] = df['Target'].astype(int)        
    
    # Limpiar todas las filas que contienen NaN (Necesario por las SMAs y el Target)
    df = df.dropna() 

    print("\n--- Vista Preliminar (Datos listos para ML) ---")
    print(df.tail()) 

except KeyError:
    print("\nError de clave: La API no devolvió los datos esperados. Revisa tu clave API o el límite de llamadas.")
    exit()
except Exception as e:
    print(f"\nOcurrió un error general durante el procesamiento de Pandas: {e}")
    exit()

# ---------------------------------------------------------------------

# --- FASE 2: ANÁLISIS DE RIESGO COMPLETO ---

volatilidad_diaria = df['daily_returns'].std() 
volatilidad_anualizada = volatilidad_diaria * np.sqrt(MARKET_DAYS)
rendimiento_anualizado = df['daily_returns'].mean() * MARKET_DAYS
ratio_sharpe = (rendimiento_anualizado - TASA_LIBRE_RIESGO) / volatilidad_anualizada

rolling_max = df['cumulative_returns'].cummax()
drawdown = (df['cumulative_returns'] / rolling_max) - 1
mdd = drawdown.min()
ratio_calmar = rendimiento_anualizado / abs(mdd)

print("\n--- INFORME FINANCIERO (FASE 2) ---")
print(f"Volatilidad Anualizada de {SYMBOL}: {volatilidad_anualizada:.2%}")
print(f"Rendimiento Anualizado de {SYMBOL}: {rendimiento_anualizado:.2%}")
print(f"Ratio de Sharpe de {SYMBOL}: {ratio_sharpe:.4f}")
print(f"Máxima Caída (MDD) de {SYMBOL}: {mdd:.2%}")
print(f"Ratio de Calmar de {SYMBOL}: {ratio_calmar:.4f}")

# ---------------------------------------------------------------------

# --- FASE 4: ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING ---

# Definición de X e Y (¡Ahora las columnas existen en df!)
FEATURES = ['close', 'SMA_20', 'SMA_50', 'Median_10', 'volume'] 
X = df[FEATURES]
Y = df['Target']

# 1. División de Datos (70% entrenamiento, 30% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# 2. Entrenamiento del Modelo
modelo = LogisticRegression(solver='lbfgs', max_iter=1000)
modelo.fit(X_train, Y_train)

# 3. Predicción y Evaluación
Y_pred = modelo.predict(X_test)
precision = accuracy_score(Y_test, Y_pred)

# --- PREDICCIÓN EN TIEMPO REAL (Último Día) ---
# Aísla la última fila con los FEATURES (datos de hoy)
X_today = df[FEATURES].iloc[-1].values.reshape(1, -1)

# Realiza la predicción
prediccion_manana = modelo.predict(X_today)[0]

print("\n--- ANÁLISIS DE MACHINE LEARNING (FASE 4) ---")
print(f"Precisión del Modelo (Accuracy): {precision:.2%}")
print("\n--- PREDICCIÓN PARA EL PRÓXIMO DÍA DE MERCADO ---")

if prediccion_manana == 1:
    print("📈 El modelo predice que el precio de cierre de mañana SUBIRÁ (Target = 1).")
else:
    print("📉 El modelo predice que el precio de cierre de mañana BAJARÁ o se MANTENDRÁ (Target = 0).")


# 5. VISUALIZACIÓN DE PRECIO, VOLUMEN Y TENDENCIAS (Matplotlib con Dashboard)

# Definición de la etiqueta de la predicción
if prediccion_manana == 1:
    prediccion_texto = "SUBIRÁ (1) 🟢"
else:
    prediccion_texto = "BAJARÁ/ESPERA (0) 🔴"
    
# Formateo de las métricas para el dashboard
dashboard_text = (
    f"--- Dashboard de Análisis ({SYMBOL}) ---\n"
    f"1. Volatilidad Anualizada: {volatilidad_anualizada:.2%}\n"
    f"2. Rendimiento Anualizado: {rendimiento_anualizado:.2%}\n"
    f"3. Ratio de Sharpe: {ratio_sharpe:.2f}\n"
    f"4. Ratio de Calmar: {ratio_calmar:.2f}\n"
    f"5. Máxima Caída (MDD): {mdd:.2%}\n"
    f"6. Precisión del Modelo (Test): {precision:.2%}\n"
    f"7. Predicción para Mañana: {prediccion_texto}"
)

# 1. Creamos una figura con dos subgráficos (2 filas, 1 columna).
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})

# --- Gráfico Superior: Precio de Cierre y Medias Móviles (ax1) ---
ax1.plot(df.index, df['close'], color='blue', label='Precio de Cierre') 
if 'SMA_20' in df.columns:
    ax1.plot(df.index, df['SMA_20'], color='orange', label='SMA 20 días', linestyle='--')
if 'SMA_50' in df.columns:
    ax1.plot(df.index, df['SMA_50'], color='red', label='SMA 50 días', linestyle='--')

ax1.set_title(f'Análisis de Precio, Volumen y Tendencias de {SYMBOL}', fontsize=16)
ax1.set_ylabel('Precio')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left') # Mueve la leyenda para no chocar con el dashboard

# --- Gráfico Inferior: Volumen (ax2) ---
if 'volume' in df.columns:
    ax2.bar(df.index, df['volume'], color='gray', alpha=0.6, label='Volumen')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Volumen')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

plt.tight_layout(rect=[0, 0, 0.80, 1]) # Deja espacio a la derecha para el texto

# --- ADICIÓN DEL DASHBOARD DE TEXTO (Cuadrado nuevo) ---
plt.figtext(0.70, 0.90, dashboard_text, 
            fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7, ec="gray"))
# *********************************************************

plt.show() 

print("\nScript ejecutado completamente con la visualización del Dashboard.")