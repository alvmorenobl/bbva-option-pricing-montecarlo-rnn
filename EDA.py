"""Análisis exploratorio de los datos en bruto del Dow Jones (Reto 9).

Lee y une los CSV originales (uno por empresa, formato marketstack), describe
cómo llegan los datos y saca los gráficos generales. Tablas en Datos/ y
gráficos en figuras/. No usa el CSV limpio.

    python EDA.py                     # lee Datos/DowJones
    python EDA.py ruta/a/los/csv      # otra carpeta
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CARPETA_ENTRADA = sys.argv[1] if len(sys.argv) > 1 else os.path.join("Datos", "DowJones")
CARPETA_DATOS = "Datos"
CARPETA_FIGURAS = "figuras"
DIAS_ANIO = 252
SALTO_REVISAR = 0.25   # rentabilidad diaria que se lista como salto
SALTO_SPLIT = 0.50     # a partir de aquí se trata como split sin ajustar y se excluye
COLUMNAS = ["open", "high", "low", "last", "close", "volume"]

# Paleta del proyecto, la de seleccioncartera.ipynb
COLORES = {
    "azul_principal": "#001391", "azul_claro": "#4D5FC4", "azul_muy_claro": "#B3BAE8",
    "gris_oscuro": "#1A1A2E", "gris_medio": "#6E7191", "gris_claro": "#E4E4EC",
    "blanco": "#FFFFFF", "acento": "#FF4B4B",
}
MAPA_CORR = LinearSegmentedColormap.from_list(
    "corr", [COLORES["acento"], COLORES["blanco"], COLORES["azul_principal"]])
MAPA_NULOS = LinearSegmentedColormap.from_list(
    "nulos", [COLORES["blanco"], COLORES["azul_claro"], COLORES["azul_principal"]])

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


def cargar(carpeta):
    """Une todos los CSV de la carpeta en una sola tabla."""
    ficheros = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
    if not ficheros:
        raise FileNotFoundError(f"No hay CSV en '{carpeta}'.")
    df = pd.concat([pd.read_csv(f) for f in ficheros], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    print(f"{len(ficheros)} ficheros -> {len(df):,} filas, {df['symbol'].nunique()} empresas, "
          f"{df['date'].min().date()} a {df['date'].max().date()}")
    print(f"Exchange: {df['exchange'].unique().tolist()}")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def calidad(df):
    """Una fila por empresa: tamaño, fechas, nulos, duplicados y cuándo cambia el formato."""
    filas = {}
    for s, g in df.groupby("symbol"):
        primer_last = g.loc[g["last"].notna(), "date"].min()
        filas[s] = {
            "filas": len(g),
            "inicio": g["date"].min().date(),
            "fin": g["date"].max().date(),
            "dias_con_datos": g["date"].dt.date.nunique(),
            "filas_antes_de_last": int((g["date"] < primer_last).sum()),
            "duplicados": int(g.duplicated("date").sum()),
            **{f"nulos_{c}_%": round(g[c].isna().mean() * 100, 1) for c in COLUMNAS},
        }
    return pd.DataFrame(filas).T.rename_axis("symbol")


def precio_diario(g):
    """Cierre diario de una empresa a partir de las filas intradía.

    Hasta que aparece `last` el dato diario está en `close`; desde entonces el
    precio real del día es `last` (el `close` intradía es el cierre del día
    anterior). Las filas de medianoche pertenecen al día anterior.
    """
    g = g.set_index("date").sort_index()
    precio = g["last"].copy()
    primero = g["last"].first_valid_index()
    antes = g.index < primero if primero is not None else np.ones(len(g), bool)
    precio[antes] = g.loc[antes, "close"]
    precio.index = precio.index - pd.Timedelta(hours=3)
    return precio.resample("B").last()


def volumen_medio(df):
    """Volumen medio diario. Solo las filas de medianoche traen el volumen total del día."""
    resumen = df[(df["date"].dt.hour == 0) & (df["date"].dt.minute == 0)]
    return resumen.groupby("symbol")["volume"].mean().reindex(sorted(df["symbol"].unique()))


# ---------------------------------------------------------------- gráficos

def grafico_cobertura(cal):
    cal = cal.sort_values(["inicio", "filas"], ascending=[False, True])
    ini = mdates.date2num(pd.to_datetime(cal["inicio"]))
    fin = mdates.date2num(pd.to_datetime(cal["fin"]))
    largo = pd.to_datetime(cal["inicio"]) < pd.Timestamp("2019-01-01")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(cal.index, fin - ini, left=ini, height=0.65,
            color=np.where(largo, COLORES["azul_principal"], COLORES["azul_muy_claro"]))
    for y, (f, x) in enumerate(zip(cal["filas"], fin)):
        ax.text(x + 8, y, f"{f:,}".replace(",", "."), va="center", fontsize=7)
    ax.xaxis_date()
    ax.set_xlim(ini.min() - 20, fin.max() + 140)
    ax.grid(axis="y", visible=False)
    ax.set_title("Cobertura temporal de cada CSV (y nº de filas)\n"
                 f"{int(largo.sum())} empresas desde 2018, {int((~largo).sum())} desde mayo-junio de 2020")
    guardar(fig, "cobertura.png")


def grafico_nulos(cal):
    m = cal[[f"nulos_{c}_%" for c in COLUMNAS]].astype(float)
    m.columns = COLUMNAS
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(m, cmap=MAPA_NULOS, vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(COLUMNAS)), COLUMNAS)
    ax.set_yticks(range(len(m)), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 50 else COLORES["gris_oscuro"])
    fig.colorbar(im, label="% de filas vacías")
    ax.set_title("Missings por columna y empresa")
    guardar(fig, "nulos.png")


def grafico_precios(P, inicio_comun, con_split):
    sin_split = [c for c in P.columns if c not in con_split]
    base = P.loc[inicio_comun:, sin_split]
    base100 = base / base.bfill().iloc[0] * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(base100.index, base100.values, color=COLORES["gris_medio"], lw=0.6, alpha=0.6)
    a1.plot(base100.index, base100.median(axis=1), color=COLORES["azul_principal"], lw=2,
            label="mediana")
    a1.set_title(f"Precio diario, base 100 en {inicio_comun.date()}\n"
                 f"({len(sin_split)} empresas, sin las que tienen split sin ajustar)")
    a1.legend()
    for s in con_split:
        a2.plot(P.index, P[s], color=COLORES["azul_principal"], lw=1.2, label=s)
        r = P[s].pct_change(fill_method=None)
        dia = r.abs().idxmax()
        a2.annotate(f"{r[dia]:.0%} el {dia.date()}", (dia, P[s][dia]),
                    xytext=(15, 25), textcoords="offset points", color=COLORES["acento"],
                    arrowprops={"arrowstyle": "->", "color": COLORES["acento"]})
    a2.set_title("Precio diario sin ajustar por splits")
    a2.legend()
    guardar(fig, "precios.png")


def grafico_volatilidad(res):
    s = res["volatilidad"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(s.index, s.values, color=COLORES["azul_principal"])
    ax.axhline(s.mean(), color=COLORES["acento"], ls="--", lw=1, label=f"media {s.mean():.0%}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="x", visible=False)
    ax.set_title("Volatilidad anualizada por empresa (periodo común, sin splits)")
    ax.legend()
    guardar(fig, "volatilidad.png")


def grafico_correlacion(corr):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(13, 11))
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
    ax.set_title("Correlación entre las 26 empresas (rentabilidades diarias, periodo común)")
    guardar(fig, "correlacion.png")


def grafico_distribucion(rent, res):
    orden = res["volatilidad"].sort_values(ascending=False).index
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})
    a1.hist(rent.stack().dropna() * 100, bins=80, range=(-10, 10), color=COLORES["azul_claro"])
    a1.set_xlabel("rentabilidad diaria (%)")
    a1.set_ylabel("nº de observaciones")
    a1.set_title("Rentabilidad diaria, todas las empresas")
    a2.boxplot([rent[c].dropna() * 100 for c in orden], tick_labels=orden, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor=COLORES["azul_muy_claro"]),
               medianprops=dict(color=COLORES["azul_principal"], lw=2))
    a2.tick_params(axis="x", rotation=90)
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("rentabilidad diaria (%)")
    a2.set_title("Dispersión por empresa (ordenadas por volatilidad)")
    guardar(fig, "distribucion_retornos.png")


def grafico_volumen(vol):
    v = (vol / 1e6).dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(v.index, v.values, color=COLORES["azul_claro"])
    ax.set_ylabel("millones de acciones")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="x", visible=False)
    ax.set_title("Volumen medio diario por empresa")
    guardar(fig, "volumen.png")


# ---------------------------------------------------------------- principal

def main():
    os.makedirs(CARPETA_DATOS, exist_ok=True)
    os.makedirs(CARPETA_FIGURAS, exist_ok=True)

    print("\n1. Unión de los CSV")
    df = cargar(CARPETA_ENTRADA)

    print("\n2. Calidad de los datos")
    cal = calidad(df)
    cal.to_csv(os.path.join(CARPETA_DATOS, "calidad_por_empresa.csv"))
    print(f"Duplicados (empresa, fecha): {cal['duplicados'].sum()}")
    print("Vacíos por columna (%):", {c: round(df[c].isna().mean() * 100, 1) for c in COLUMNAS})
    print(f"Empresas con datos desde 2018: {(pd.to_datetime(cal['inicio']) < '2019-01-01').sum()}")
    grafico_cobertura(cal)
    grafico_nulos(cal)

    print("\n3. Serie diaria")
    P = pd.DataFrame({s: precio_diario(g) for s, g in df.groupby("symbol")})
    P = P.dropna(how="all")
    rent = P.pct_change(fill_method=None)
    saltos = rent.stack()
    saltos = saltos[saltos.abs() >= SALTO_REVISAR].rename("rentabilidad").reset_index()
    saltos.columns = ["fecha", "symbol", "rentabilidad"]
    saltos["fecha"] = saltos["fecha"].dt.date
    saltos["posible_split"] = saltos["rentabilidad"].abs() >= SALTO_SPLIT
    saltos.to_csv(os.path.join(CARPETA_DATOS, "saltos_precio.csv"), index=False)
    print(f"Días con salto >= {SALTO_REVISAR:.0%}: {len(saltos)}")
    print(saltos.to_string(index=False))
    con_split = saltos.loc[saltos["posible_split"], "symbol"].unique().tolist()
    rent = rent.mask(rent.abs() >= SALTO_SPLIT)   # los splits no son rentabilidad

    inicio_comun = P.apply(lambda c: c.first_valid_index()).max()
    comun = rent.loc[inicio_comun:]
    print(f"Periodo común a las {P.shape[1]} empresas: {inicio_comun.date()} a "
          f"{P.index.max().date()} ({len(comun)} días)")
    grafico_precios(P, inicio_comun, con_split)

    print("\n4. Rentabilidad y riesgo (periodo común)")
    z = comun.stack()
    res = pd.DataFrame({
        "rentabilidad": comun.mean() * DIAS_ANIO,
        "volatilidad": comun.std() * np.sqrt(DIAS_ANIO),
        "asimetria": comun.skew(),
        "curtosis_exceso": comun.kurt(),
    })
    n = comun.count()
    jb = n / 6 * (res["asimetria"] ** 2 + res["curtosis_exceso"] ** 2 / 4)
    res["jarque_bera_p"] = np.exp(-jb / 2)       # chi-cuadrado con 2 grados de libertad
    res = res.sort_values("volatilidad", ascending=False).round(4)
    res.to_csv(os.path.join(CARPETA_DATOS, "resumen_empresas.csv"))
    print(res.to_string())
    print(f"Rechazan normalidad (p<0,05): {(res['jarque_bera_p'] < 0.05).sum()} de {len(res)}")
    corr = comun.corr()
    corr.to_csv(os.path.join(CARPETA_DATOS, "matriz_correlacion.csv"))
    pares = corr.values[np.triu_indices(len(corr), 1)]
    print(f"Correlación entre pares: media {pares.mean():.2f}, mín {pares.min():.2f}, "
          f"máx {pares.max():.2f}")
    grafico_volatilidad(res)
    grafico_correlacion(corr)
    grafico_distribucion(comun, res)

    print("\n5. Volumen")
    vol = volumen_medio(df)
    grafico_volumen(vol)


if __name__ == "__main__":
    main()
