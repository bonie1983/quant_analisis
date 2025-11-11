import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 1. CONFIGURACIÓN: REEMPLAZA CON TU CLAVE
API_KEY = "TU_CLAVE_API" # <-- ¡IMPORTANTE! Reemplaza esto
SYMBOL = "BTC"
MARKET = "USD"
FUNCTION = "DIGITAL_CURRENCY_DAILY"

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

# 4. CONVERSIÓN Y LIMPIEZA CON PANDAS (CON CORRECCIÓN DE COLUMNAS A 5)
try:
    key_prices = 'Time Series (Digital Currency Daily)'
    
    # JSON a DataFrame y transposición (.T)
    df = pd.DataFrame(data[key_prices]).T

    # *** CORRECCIÓN DEL ERROR DE LONGITUD ***
    # La lista de nombres debe tener la misma longitud que el DataFrame (5 elementos)
    COLUMNAS_FINALES = ['open (USD)', 'high (USD)', 'low (USD)', 'close (USD)', 
                        'volume (USD)'] # Quitamos 'market_cap (USD)'
    
    # Comprobación de longitud antes de asignar
    if len(COLUMNAS_FINALES) != df.shape[1]:
        print("\n--- ERROR DETECTADO ---")
        print(f"ERROR: Se detectaron {df.shape[1]} columnas, pero la lista de nombres tiene {len(COLUMNAS_FINALES)}.")
        print(f"Nombres originales (para referencia): {df.columns.tolist()}")
        print("--- REVISAR LA API KEY y el JSON ---")
        exit()

    # Asignamos los 5 nombres
    df.columns = COLUMNAS_FINALES
    
    # Limpieza de Índices y Tipos de Datos
    df.index = pd.to_datetime(df.index) 
    
    # Convertir las columnas numéricas a tipo flotante
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9\.]', ''), errors='coerce')

    # Agregar columna de retornos diarios
    df = df.sort_index(ascending=True)
    df['daily_returns'] = df['close (USD)'].pct_change()

    # Calcular retornos acumulativos
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()

    # Calcular retornos logarítmicos
    df['log_returns'] = np.log(df['close (USD)'] / df['close (USD)'].shift(1))
    
    print("\n--- Vista Preliminar (Últimas 5 Filas Limpias y con retornos) ---")
    print(df.head()) 

except KeyError:
    print("\nError de clave: La API no devolvió los datos esperados. Revisa tu clave API.")
    exit()
except Exception as e:
    print(f"\nOcurrió un error general durante el procesamiento de Pandas: {e}")
    exit()

# CALCULO DE VOLATILIDAD

TASA_LIBRE_RIESGO = 0.04  # 4% anual como ejemplo

# Volatilidad diaria
volatilidad_diaria = df['daily_returns'].std() 
# Volatilidad anualizada (asumiendo 252 días de trading para acciones)
volatilidad_anualizada = volatilidad_diaria * np.sqrt(252)
print(f"Volatilidad Anualizada de {SYMBOL}: {volatilidad_anualizada:.2%}")

# CALCULO RENDIMIENTO ANUALIZADO
rendimiento_anualizado = df['daily_returns'].mean() * 365
print(f"Rendimiento Anualizado de {SYMBOL}: {rendimiento_anualizado:.2%}")

 # CALCULO RATIO SHARPE
ratio_sharpe = (rendimiento_anualizado - TASA_LIBRE_RIESGO) / volatilidad_anualizada
print(f"Ratio de Sharpe de {SYMBOL}: {ratio_sharpe:.4f}")

# CALCULO MAXIMA CAIDA (MDD)
rolling_max = df['cumulative_returns'].cummax() # 1. Encuentra el pico más alto hasta la fecha
drawdown = (df['cumulative_returns'] / rolling_max) - 1 # 2. Mide la caída desde el pico
mdd = drawdown.min() # 3. Encuentra la máxima caída histórica
print(f"Máxima Caída (MDD) de {SYMBOL}: {mdd:.2%}")

# CALCULO RATIO CALMAR
ratio_calmar = rendimiento_anualizado / abs(mdd) # Para calcular el Ratio de Calmar hay que usar el valor absoluto de MDD en positivo
print(f"Ratio de Calmar de {SYMBOL}: {ratio_calmar:.4f}")

# 5. PRIMERAS VISUALIZACIONES (Matplotlib con Subgráficos)

# 1. Creamos una figura con dos subgráficos (2 filas, 1 columna).
# Usamos sharex=True para que el eje X (Fechas) sea común y se muestren alineados.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]}) # Asigna más espacio al precio

# --- Gráfico Superior: Precio de Cierre (ax1) ---
ax1.plot(df.index, df['close (USD)'], color='blue', label='Precio de Cierre')
ax1.set_title(f'Análisis de Precio y Volumen de {SYMBOL}', fontsize=14)
ax1.set_ylabel('Precio (USD)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- Gráfico Inferior: Volumen (ax2) ---
# Usamos gráfico de barras (bar) para representar el volumen
ax2.bar(df.index, df['volume (USD)'], color='gray', alpha=0.6, label='Volumen')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Volumen (USD)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# Ajusta el espacio entre subgráficos
plt.tight_layout()
plt.show() 

print("\nScript ejecutado correctamente. ¡Ahora con Precio y Volumen!")