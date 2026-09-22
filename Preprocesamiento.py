"""
Preprocesamiento.py

Dataset de entrada: Datos/DowJones/*.csv (26 ficheros, uno por acción, formato marketstack)
Ruta dentro del repositorio: Datos/DowJones/

Objetivo: dejar un dataset único, limpio, con frecuencia homogénea y sin missings,
que sirva de entrada para la selección de la cartera (Sección 3 de la descripción técnica).

Pasos:
1. Carga y unificación
2. Duplicados y orden temporal
3. Completado con Yahoo Finance (símbolos que faltan del Dow 30)
4. Comprobación de calidad / coherencia OHLC (aplicada a TODO el dataset, incluidos los datos de yfinance)
5. Homogenización de la frecuencia muestral (solo a los 26 símbolos intradía; los de yfinance ya vienen diarios)
6. Tratamiento de missings (aplicado a TODO el dataset, incluidos los datos de yfinance)
7. Análisis descriptivo
8. Guardado del dataset limpio
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

pd.set_option("display.max_columns", None)


# ---------------------------------------------------------------------------
# Funciones reutilizables (se aplican tanto a los datos originales como a los
# descargados de yfinance, para que ambos reciban el mismo tratamiento)
# ---------------------------------------------------------------------------

def marcar_incoherencias_ohlc(df):
    """Marca como NaN los precios financieramente incoherentes:
    high < low, precios <= 0, o el open/close fuera del rango [low, high]."""
    cond_incoherente = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"]) | (df["high"] < df["close"])
        | (df["low"] > df["open"])  | (df["low"] > df["close"])
        | (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    )
    cols_precio = ["open", "high", "low", "close", "last"]
    df.loc[cond_incoherente, cols_precio] = np.nan
    return df, int(cond_incoherente.sum())


def imputar_missings(df):
    """LOCF (con bfill de respaldo solo para el primer registro de cada serie)
    para los precios; el volumen en un día sin sesión se rellena con 0."""
    for col in ["open", "high", "low", "last", "close"]:
        df[col] = df.groupby("symbol")[col].transform(lambda x: x.ffill().bfill())
    df["volume"] = df["volume"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 1. Carga y unificación de los datos
# ---------------------------------------------------------------------------

RUTA_DATOS = os.path.join("Datos", "DowJones")

lista_df = []
for nombre_archivo in os.listdir(RUTA_DATOS):
    if nombre_archivo.endswith(".csv"):
        temp = pd.read_csv(os.path.join(RUTA_DATOS, nombre_archivo))
        lista_df.append(temp)

df = pd.concat(lista_df, ignore_index=True)
df["date"] = pd.to_datetime(df["date"], utc=True)

print(f"Archivos leidos: {len(lista_df)}")
print(f"Filas totales:   {len(df)}")
print(f"Simbolos:        {df['symbol'].nunique()}")


# ---------------------------------------------------------------------------
# 2. Duplicados y orden temporal
# ---------------------------------------------------------------------------

df = df.drop_duplicates(subset=["symbol", "date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
print("Filas tras quitar duplicados:", len(df))


# ---------------------------------------------------------------------------
# 3. Comprobación de calidad / coherencia OHLC (sobre los datos intradía)
# ---------------------------------------------------------------------------

df, n_incoherentes = marcar_incoherencias_ohlc(df)
print("Filas con incoherencias OHLC:", n_incoherentes)


# ---------------------------------------------------------------------------
# 4. Homogenización de la frecuencia muestral (a diario, dias habiles)
# ---------------------------------------------------------------------------

def resample_symbol(g):
    g = g.set_index("date").sort_index()
    return pd.DataFrame({
        "open":   g["open"].resample("B").first(),
        "high":   g["high"].resample("B").max(),
        "low":    g["low"].resample("B").min(),
        "last":   g["last"].resample("B").last(),
        "close":  g["close"].resample("B").last(),
        "volume": g["volume"].resample("B").sum(),
    })

daily_list = []
for symbol, g in df.groupby("symbol"):
    d = resample_symbol(g)
    d["symbol"] = symbol
    d["exchange"] = g["exchange"].iloc[0]
    daily_list.append(d)

df_daily = pd.concat(daily_list).reset_index()
print("Filas tras homogenizar a frecuencia diaria:", len(df_daily))


# ---------------------------------------------------------------------------
# 5. Completado con datos de Yahoo Finance (símbolos que faltan del Dow 30)
#    NOTA: esta celda necesita conexión a internet para ejecutarse.
# ---------------------------------------------------------------------------

TICKERS_DOW30 = {
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS", "HD",
    "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
    "TRV", "UNH", "V", "VZ", "WBA", "WMT",
}
simbolos_actuales = set(df_daily["symbol"].unique())
simbolos_faltantes = sorted(TICKERS_DOW30 - simbolos_actuales)
print("Simbolos que faltan y se completan con yfinance:", simbolos_faltantes)

fecha_ini = df_daily["date"].min().date()
fecha_fin = df_daily["date"].max().date()

nuevos = []
for simbolo in simbolos_faltantes:
    hist = yf.download(simbolo, start=fecha_ini, end=fecha_fin, progress=False, auto_adjust=False)
    if hist.empty:
        print(f"Aviso: no se ha podido descargar {simbolo}, se omite.")
        continue
    if isinstance(hist.columns, pd.MultiIndex):
        # yfinance moderno devuelve columnas (Precio, Ticker) al descargar 1 solo simbolo
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    hist["last"] = hist["close"]
    hist["symbol"] = simbolo
    hist["exchange"] = "YFINANCE"
    hist["date"] = pd.to_datetime(hist["date"], utc=True)
    nuevos.append(hist[["open", "high", "low", "last", "close", "volume", "date", "symbol", "exchange"]])

if nuevos:
    df_daily = pd.concat([df_daily] + nuevos, ignore_index=True)
    df_daily = df_daily.sort_values(["symbol", "date"]).reset_index(drop=True)

print("Filas totales tras completar con yfinance:", len(df_daily))
print("Simbolos totales:", df_daily["symbol"].nunique())

# Repetimos la comprobación de coherencia OHLC sobre el dataset completo, para
# que los datos recién llegados de yfinance pasen por el mismo control de calidad
df_daily, n_incoherentes_yf = marcar_incoherencias_ohlc(df_daily)
print("Filas con incoherencias OHLC (tras incorporar yfinance):", n_incoherentes_yf)


# ---------------------------------------------------------------------------
# 6. Tratamiento de missings (sobre el dataset ya completo, 30 simbolos)
# ---------------------------------------------------------------------------

print("Missings antes de imputar:")
print(df_daily.isna().sum())

df_daily = imputar_missings(df_daily)

print("\nMissings despues de imputar (debe ser todo 0):")
print(df_daily.isna().sum())
assert df_daily[["open", "high", "low", "last", "close", "volume"]].isna().sum().sum() == 0, \
    "Quedan missings sin tratar"


# ---------------------------------------------------------------------------
# 7. Análisis descriptivo
# ---------------------------------------------------------------------------

print(df_daily.groupby("symbol")["close"].describe())

fig, ax = plt.subplots(figsize=(14, 6))
for symbol, g in df_daily.groupby("symbol"):
    ax.plot(g["date"], g["close"], label=symbol, linewidth=1)
ax.set_title("Evolucion del precio de cierre (todas las acciones)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Close")
ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join("Datos", "evolucion_precios.png"))
plt.show()

retornos = df_daily.pivot(index="date", columns="symbol", values="close").pct_change()
volatilidad = retornos.std().sort_values(ascending=False)
volatilidad.plot(kind="bar", figsize=(12, 4), title="Volatilidad diaria por accion")
plt.tight_layout()
plt.savefig(os.path.join("Datos", "volatilidad.png"))
plt.show()


# ---------------------------------------------------------------------------
# 8. Guardado del dataset limpio
# ---------------------------------------------------------------------------

RUTA_SALIDA = os.path.join("Datos", "dowjones_limpio.csv")
df_daily.to_csv(RUTA_SALIDA, index=False)
print("Guardado en:", RUTA_SALIDA)