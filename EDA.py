"""Análisis exploratorio del Dow Jones (Reto 9).

Lee Datos/dowjones_limpio.csv, guarda tablas en Datos/ y gráficos en figuras/.
Los retornos son simples y diarios sobre adj_close, como en seleccioncartera.ipynb.

    python EDA.py                 # trío por defecto: WMT, MRK, NVDA
    python EDA.py JNJ WMT NVDA    # otro trío
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, ScalarFormatter

RUTA_ENTRADA = os.path.join("Datos", "dowjones_limpio.csv")
CARPETA_DATOS = "Datos"
CARPETA_FIGURAS = "figuras"
TRIO_POR_DEFECTO = ["WMT", "MRK", "NVDA"]
DIAS_ANIO = 252
VENTANAS_VOL = (60, 252)
VENTANA_CORR = 252
SUBPERIODOS = {"2018-2019": ("2018", "2019"), "2020": ("2020", "2020"),
               "2021-2022": ("2021", "2022"), "2023-2025": ("2023", "2025")}

# Paleta del proyecto, la de seleccioncartera.ipynb
COLORES = {
    "azul_principal": "#001391", "azul_claro": "#4D5FC4", "azul_muy_claro": "#B3BAE8",
    "gris_oscuro": "#1A1A2E", "gris_medio": "#6E7191", "gris_claro": "#E4E4EC",
    "blanco": "#FFFFFF", "acento": "#FF4B4B",
}
MAPA_CORR = LinearSegmentedColormap.from_list(
    "corr", [COLORES["acento"], COLORES["blanco"], COLORES["azul_principal"]])
COLORES_TRIO = [COLORES["azul_principal"], COLORES["azul_claro"], COLORES["gris_medio"]]

plt.rcParams.update({
    "axes.edgecolor": COLORES["gris_medio"], "axes.labelcolor": COLORES["gris_oscuro"],
    "axes.titlecolor": COLORES["gris_oscuro"], "text.color": COLORES["gris_oscuro"],
    "xtick.color": COLORES["gris_oscuro"], "ytick.color": COLORES["gris_oscuro"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": COLORES["gris_claro"],
    "legend.frameon": False,
})


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_FIGURAS, nombre)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figura:", ruta)


def cargar(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Falta '{ruta}'. Genera antes el CSV limpio "
                                "(Preprocesamiento_limpieza.py, rama limpieza).")
    df = pd.read_csv(ruta)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    nulos = df[["open", "high", "low", "close", "adj_close", "volume"]].isna().sum().sum()
    duplicados = df.duplicated(["symbol", "date"]).sum()
    print(f"Cargado '{ruta}': {len(df):,} filas, {df['symbol'].nunique()} símbolos, "
          f"{df['date'].min().date()} a {df['date'].max().date()}")
    print(f"Nulos: {nulos}, duplicados symbol+date: {duplicados}")
    if nulos or duplicados:
        print("AVISO: el CSV no está tan limpio como debería, revisa la limpieza.")
    return df


def barras(serie, titulo, ylabel, archivo, trio):
    colores = [COLORES["acento"] if s in trio else COLORES["azul_claro"] for s in serie.index]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(serie.index, serie.values, color=colores)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{titulo}\n(en rojo: {' + '.join(trio)})")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="x", visible=False)
    guardar(fig, archivo)


def matriz(corr):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap=MAPA_CORR, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > 0.7 else COLORES["gris_oscuro"])
    fig.colorbar(im, shrink=0.8)
    ax.set_title(f"Correlación entre las {n} acciones (rentabilidades diarias)")
    guardar(fig, "matriz_correlacion.png")


def riesgo_rentabilidad(resumen, trio):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for simbolo, fila in resumen.iterrows():
        x, y = fila["volatilidad_anualizada"], fila["rentabilidad_anualizada"]
        en_trio = simbolo in trio
        ax.scatter(x, y, s=90 if en_trio else 40, zorder=3,
                   color=COLORES["acento"] if en_trio else COLORES["azul_claro"])
        ax.annotate(simbolo, (x, y), xytext=(5, 4), textcoords="offset points",
                    fontsize=9 if en_trio else 7,
                    fontweight="bold" if en_trio else "normal",
                    color=COLORES["gris_oscuro"] if en_trio else COLORES["gris_medio"])
    ax.set_xlabel("Volatilidad anualizada")
    ax.set_ylabel("Rentabilidad anualizada")
    ax.set_title(f"Riesgo frente a rentabilidad ({len(resumen)} acciones)\n"
                 f"(en rojo: {' + '.join(trio)})")
    guardar(fig, "riesgo_rentabilidad.png")


def precios_base100(precios, trio):
    base100 = precios / precios.iloc[0] * 100
    resto = [s for s in base100.columns if s not in trio]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(base100.index, base100[resto], color=COLORES["azul_muy_claro"], lw=0.9)
    ax.plot([], [], color=COLORES["azul_muy_claro"], label="Resto de acciones")
    for simbolo, color in zip(trio, COLORES_TRIO):
        ax.plot(base100.index, base100[simbolo], color=color, lw=2.2, label=simbolo)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Precio ajustado (base 100, escala logarítmica)")
    ax.set_title(f"Evolución de precios ({base100.shape[1]} acciones, base 100)")
    ax.legend(ncol=4)
    guardar(fig, "precios_base100.png")


def correlacion_media(corr, trio):
    n = len(corr)
    media = ((corr.sum() - 1) / (n - 1)).sort_values()
    colores = [COLORES["acento"] if s in trio else COLORES["azul_claro"] for s in media.index]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(media.index, media.values, color=colores)
    ax.set_ylabel("Correlación media")
    ax.set_title(f"Correlación media de cada acción con las otras {n - 1}\n"
                 f"(en rojo: {' + '.join(trio)})")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="x", visible=False)
    guardar(fig, "correlacion_media.png")


def volatilidad_movil(rent, trio):
    fig, axes = plt.subplots(len(VENTANAS_VOL), 1, figsize=(12, 7), sharex=True)
    for ax, ventana in zip(axes, VENTANAS_VOL):
        vol = rent[trio].rolling(ventana).std() * np.sqrt(DIAS_ANIO)
        for simbolo, color in zip(trio, COLORES_TRIO):
            ax.plot(vol.index, vol[simbolo], color=color, lw=1.5, label=simbolo)
        ax.set_title(f"Volatilidad anualizada móvil ({ventana} días)")
        ax.set_ylabel("Volatilidad")
    axes[0].legend(ncol=len(trio))
    guardar(fig, "estabilidad_volatilidad_trio.png")


def correlacion_movil(rent, trio):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for (a, b), color in zip(combinations(trio, 2), COLORES_TRIO):
        corr = rent[a].rolling(VENTANA_CORR).corr(rent[b])
        ax.plot(corr.index, corr, color=color, lw=1.5, label=f"{a} - {b}")
    ax.set_title(f"Correlación móvil ({VENTANA_CORR} días)")
    ax.set_ylabel("Correlación")
    ax.legend(ncol=3)
    guardar(fig, "estabilidad_correlacion_trio.png")


def estabilidad(rent, trio):
    pares = list(combinations(trio, 2))
    filas = {}
    for nombre, (ini, fin) in SUBPERIODOS.items():
        r = rent[trio].loc[ini:fin]
        corr = r.corr()
        fila = {f"vol_{s}": r[s].std() * np.sqrt(DIAS_ANIO) for s in trio}
        fila.update({f"corr_{a}_{b}": corr.loc[a, b] for a, b in pares})
        fila["corr_media"] = np.mean([corr.loc[a, b] for a, b in pares])
        filas[nombre] = fila
    return pd.DataFrame(filas).T.rename_axis("periodo")


def main():
    trio = [s.upper() for s in sys.argv[1:]] or TRIO_POR_DEFECTO
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    df = cargar(RUTA_ENTRADA)
    faltan = [s for s in trio if s not in set(df["symbol"])]
    if faltan:
        raise ValueError(f"Símbolos que no están en el dataset: {faltan}")

    precios = df.pivot(index="date", columns="symbol", values="adj_close")
    rent = precios.pct_change().iloc[1:]
    resumen = pd.DataFrame({
        "rentabilidad_anualizada": rent.mean() * DIAS_ANIO,
        "volatilidad_anualizada": rent.std() * np.sqrt(DIAS_ANIO),
        "volumen_medio": df.groupby("symbol")["volume"].mean(),
    }).sort_values("volatilidad_anualizada", ascending=False)
    corr = rent.corr()
    corr_trio = rent[trio].corr()
    estab = estabilidad(rent, trio)

    print(f"\nTrío: {' + '.join(trio)}\n")
    print(resumen.round(4).to_string())
    print("\nEstabilidad por subperíodos\n", estab.round(3).to_string())
    print("\nCorrelación del trío (para Cholesky)\n", corr_trio.round(3).to_string(), "\n")

    for nombre, tabla in {"resumen_acciones.csv": resumen, "matriz_correlacion.csv": corr,
                          "matriz_correlacion_trio.csv": corr_trio,
                          "estabilidad_trio.csv": estab}.items():
        tabla.to_csv(os.path.join(CARPETA_DATOS, nombre))
        print("  tabla: ", os.path.join(CARPETA_DATOS, nombre))

    barras(resumen["volatilidad_anualizada"].sort_values(ascending=False),
           "Volatilidad anualizada por acción", "Volatilidad anualizada",
           "ranking_volatilidad_anualizada.png", trio)
    barras(resumen["volumen_medio"].sort_values(ascending=False),
           "Volumen medio diario por acción", "Volumen medio", "volumen_medio.png", trio)
    matriz(corr)
    riesgo_rentabilidad(resumen, trio)
    precios_base100(precios, trio)
    correlacion_media(corr, trio)
    volatilidad_movil(rent, trio)
    correlacion_movil(rent, trio)
    print("\nEDA terminado.")


if __name__ == "__main__":
    main()
