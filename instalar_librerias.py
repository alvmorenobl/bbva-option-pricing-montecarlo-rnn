"""Instala de una vez todas las librerías que usan los notebooks y scripts del Reto 9.

    python instalar_librerias.py

Comprueba cada librería y solo instala con pip las que faltan. Al final imprime
la versión de cada una. Necesita Python 3.10 o superior.

Qué usa cada una:
    numpy, pandas, matplotlib  -> todos los notebooks y EDA.py
    scipy                      -> fórmula cerrada (norm), splines y GARCH propio
    statsmodels                -> ADF, KPSS, ARCH-LM, Ljung-Box, descomposición
    arch                       -> validación del GARCH propio (Analisis_series_volatilidad_final)
    plotly                     -> gráficos interactivos HTML/JS de figuras/
    yfinance                   -> descarga de Yahoo Finance (limpieza y sectores)
    torch                      -> red neuronal (rnn.ipynb y resultados.ipynb)
    jupyter                    -> abrir y ejecutar los notebooks
"""
import importlib
import subprocess
import sys

# nombre al importar -> nombre en pip
LIBRERIAS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "arch": "arch",
    "plotly": "plotly",
    "yfinance": "yfinance",
    "torch": "torch",
    "notebook": "jupyter",
}


def falta(modulo):
    try:
        importlib.import_module(modulo)
        return False
    except ImportError:
        return True


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit(f"Hace falta Python 3.10 o superior (tienes {sys.version.split()[0]}).")

    pendientes = [pip for mod, pip in LIBRERIAS.items() if falta(mod)]
    if pendientes:
        print("Instalando:", ", ".join(pendientes))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pendientes])
    else:
        print("Todas las librerías ya estaban instaladas.")

    importlib.invalidate_caches()
    print("\nVersiones:")
    for mod in LIBRERIAS:
        version = getattr(importlib.import_module(mod), "__version__", "?")
        print(f"  {mod:<12} {version}")
