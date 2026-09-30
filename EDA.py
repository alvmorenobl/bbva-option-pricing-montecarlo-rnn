import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

RUTA_ENTRADA = os.path.join("Datos", "dowjones_limpio.csv")

CARPETA_SALIDA = os.path.dirname(RUTA_ENTRADA) or "."
os.makedirs(CARPETA_SALIDA, exist_ok=True)

RUTA_VOLATILIDAD = os.path.join(CARPETA_SALIDA, "volatilidad_anualizada.csv")
RUTA_CORRELACION = os.path.join(CARPETA_SALIDA, "matriz_correlacion.csv")

DIAS_BOLSA_ANIO = 252

plt.rcParams["figure.figsize"] = (12, 5)
sns.set_style("whitegrid")

def cargar_datos_limpios(ruta: str) -> pd.DataFrame:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encuentra '{ruta}'."
        )
 
    df = pd.read_csv(ruta)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
 
    print(f"✅ Cargado '{ruta}'")
    print(f"✅ Filas: {len(df):,}  |  Símbolos: {df['symbol'].nunique()}")
    print(f"✅ Rango de fechas: {df['date'].min().date()} -> "
          f"{df['date'].max().date()}")

    n_nulos = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
    n_dup = df.duplicated(subset=["symbol", "date"]).sum()
    print(f"✅ Nulos restantes: {n_nulos}  |  Duplicados symbol+date: {n_dup}")
    if n_nulos > 0 or n_dup > 0:
        print("⚠️  El dataset de entrada no está tan limpio como se esperaba, "
              "revisa Preprocesamiento_limpieza.ipynb antes de continuar.")
 
    return df

def calcular_retornos_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = df.groupby("symbol")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    return df
 
 
def graficar_distribucion_retornos(df: pd.DataFrame, simbolo: str) -> None:
    datos = df[df["symbol"] == simbolo]["log_return"].dropna()
    mu, std = stats.norm.fit(datos)
 
    plt.figure()
    plt.hist(datos, bins=40, density=True, alpha=0.6,
              color="seagreen", edgecolor="black")
    x = np.linspace(datos.min(), datos.max(), 200)
    plt.plot(x, stats.norm.pdf(x, mu, std), "r-", linewidth=2,
              label="Ajuste Normal")
    plt.title(f"Distribución de retornos logarítmicos - {simbolo}\n"
              f"μ diario={mu:.5f}, σ diario={std:.5f}")
    plt.xlabel("Retorno logarítmico diario")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"distribucion_retornos_{simbolo}.png", dpi=150)
    plt.show()

def ranking_volatilidad(df: pd.DataFrame) -> pd.Series:
    vol = (
        df.groupby("symbol")["log_return"].std() * np.sqrt(DIAS_BOLSA_ANIO)
    ).sort_values(ascending=False)
 
    print("\n--- Volatilidad anualizada por acción (basada en retornos log) ---")
    print((vol * 100).round(2).astype(str) + " %")
 
    plt.figure(figsize=(14, 5))
    vol.plot(kind="bar", color="steelblue")
    plt.title("Volatilidad anualizada por acción (retornos logarítmicos)")
    plt.ylabel("Volatilidad anualizada")
    plt.tight_layout()
    plt.savefig("ranking_volatilidad_anualizada.png", dpi=150)
    plt.show()
 
    return vol

def rentabilidad_media_anualizada(df: pd.DataFrame) -> pd.Series:
    ret_medio = (
        df.groupby("symbol")["log_return"].mean() * DIAS_BOLSA_ANIO
    ).sort_values(ascending=False)
 
    print("\n--- Rentabilidad media anualizada por acción ---")
    print((ret_medio * 100).round(2).astype(str) + " %")
    return ret_medio

def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="date", columns="symbol", values="log_return")
    corr = pivot.corr()
 
    plt.figure(figsize=(13, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
                square=True, linewidths=0.2)
    plt.title("Matriz de correlación de retornos logarítmicos (30 acciones)")
    plt.tight_layout()
    plt.savefig("matriz_correlacion.png", dpi=150)
    plt.show()
 
    return corr

def analizar_volumen(df: pd.DataFrame) -> pd.Series:
    vol_medio = df.groupby("symbol")["volume"].mean().sort_values(
        ascending=False
    )
    print("\n--- Volumen medio diario por acción (liquidez) ---")
    print(vol_medio.round(0))
 
    plt.figure(figsize=(14, 5))
    vol_medio.plot(kind="bar", color="darkorange")
    plt.title("Volumen medio diario por acción")
    plt.ylabel("Volumen medio")
    plt.tight_layout()
    plt.savefig("volumen_medio.png", dpi=150)
    plt.show()
    return vol_medio

def guardar_resultados(vol: pd.Series, corr: pd.DataFrame) -> None:
    vol.to_frame(name="volatilidad_anualizada").to_csv(RUTA_VOLATILIDAD)
    corr.to_csv(RUTA_CORRELACION)
    print(f"\n✅ Guardado: {RUTA_VOLATILIDAD}")
    print(f"✅ Guardado: {RUTA_CORRELACION}")

if __name__ == "__main__":
    df = cargar_datos_limpios(RUTA_ENTRADA)
    df = calcular_retornos_log(df)

    simbolo_ejemplo = df["symbol"].unique()[0]
    graficar_distribucion_retornos(df, simbolo_ejemplo)
 
    vol = ranking_volatilidad(df)
    ret_medio = rentabilidad_media_anualizada(df)
    corr = matriz_correlacion(df)
    volumen = analizar_volumen(df)
 
    guardar_resultados(vol, corr)
 
    print("\n✅ Gráficos generados con éxito!")