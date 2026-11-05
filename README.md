# Reto 9 · Grupo Verde Claro · Mondragón–BBVA

Valoración de una **call asiática geométrica** sobre una cesta de tres acciones del Dow Jones (**WMT**, **MRK** y **NVDA**) con tres métodos: simulación de **Montecarlo**, la **fórmula cerrada** de BBVA y **redes neuronales recurrentes** (RNN, LSTM y GRU).

**Repositorio:** https://github.com/marioabajas/reto09-verde_claro

## Cómo ejecutarlo

1. Instalar las librerías (Python 3.10 o superior). El script solo instala las que falten:

   ```
   python instalar_librerias.py
   ```

2. Copiar las carpetas **Datos/** y **modelos/** del Drive del equipo (**R9_Entrega_EQUIPO**) en la raíz del repositorio. Ni los datos ni las redes entrenadas se suben a GitHub, tal como piden las instrucciones de entrega.
3. Ejecutar los notebooks **en este orden**, porque cada uno usa la salida del anterior. Todos leen y escriben con rutas relativas a la raíz del repositorio.

| Paso | Archivo | Entrada | Salida |
|---|---|---|---|
| 1 | **Preprocesamiento_limpieza.ipynb** | Datos/DowJones/*.csv (26 CSV originales) + Yahoo Finance | Datos/dowjones_limpio.csv |
| 2 | **EDA.py** | Datos/DowJones/*.csv | Tablas de calidad, saltos y correlación en Datos/ y gráficos en figuras/ |
| 3 | **seleccioncartera.ipynb** | Datos/dowjones_limpio.csv | Cartera WMT + MRK + NVDA, volatilidades σ y correlaciones ρ |
| 4 | **Analisis_series_volatilidad_final.ipynb** | Datos/dowjones_limpio.csv | Estacionariedad, ARCH/GARCH, Datos/volatilidad_cartera.csv, Datos/correlaciones_cartera.csv y gráficos interactivos en figuras/ |
| 5 | **montecarlo.ipynb** | σ y ρ del paso 3 (copiadas en el notebook) | Datos/resultados_montecarlo.npz (100.000 × 20 × 3 trayectorias y su payoff) |
| 6 | **rnn.ipynb** | Datos/resultados_montecarlo.npz | Redes entrenadas en modelos/ y modelos/historial.json |
| 7 | **resultados.ipynb** | Datos/resultados_montecarlo.npz, modelos/ y MonteCarlo_y_Formula_cerrada_25_26.ipynb | Comparación de primas: Montecarlo, red y fórmula cerrada |
| 8 | **figuras_informe_red.py** | modelos/, modelos/historial.json y resultados.ipynb | Las 6 figuras de la red con el formato del informe, en figuras/informe/ |

**Notas**

- Los pasos 1 y 3 necesitan conexión a internet para consultar Yahoo Finance. En el paso 3, los sectores ya están copiados en un diccionario fijo, así que el resultado no depende de Yahoo.
- **rnn.ipynb** entrena todas las redes desde cero y tarda unas tres horas y media. Con las redes ya entrenadas de **modelos/** (en el Drive), **resultados.ipynb** y **figuras_informe_red.py** se pueden ejecutar directamente.
- Todos los gráficos que dibujan los notebooks se guardan en **figuras/** al ejecutarlos. Los gráficos interactivos del análisis de series están en **figuras/index.html**.
- **MonteCarlo_y_Formula_cerrada_25_26.ipynb** es el notebook facilitado por los profesores. **resultados.ipynb** lee de él las funciones de la fórmula cerrada sin modificarlas.
- La simulación es reproducible: el comienzo común de 5 trimestres usa la semilla 11 y el reparto train, validación y test de la red usa la semilla 0.

## Estructura

```
reto09-verde_claro/
├── Datos/                    # no se sube a GitHub; está en el Drive del equipo
│   ├── DowJones/             # 26 CSV originales (formato marketstack)
│   └── ...                   # datos generados por los notebooks
├── figuras/                  # gráficos PNG de los notebooks y gráficos HTML/JS interactivos (index.html)
│   └── informe/              # figuras de la red con el formato del informe
├── modelos/                  # no se sube a GitHub; redes entrenadas (.pt) e historial, en el Drive
├── Preprocesamiento_limpieza.ipynb
├── EDA.py
├── seleccioncartera.ipynb
├── Analisis_series_volatilidad_final.ipynb
├── montecarlo.ipynb
├── rnn.ipynb
├── resultados.ipynb
├── MonteCarlo_y_Formula_cerrada_25_26.ipynb
├── figuras_informe_red.py
├── instalar_librerias.py
└── entorno_verdeclaro_reto09.yml
```

## Resumen de resultados

- **Cartera:** WMT + MRK + NVDA, la primera de 780 tríos candidatos (dos acciones estables y una volátil, sectores distintos y correlación media ≤ 0,40).
- **Montecarlo:** prima de 0,3014 ± 0,0029 (IC al 95 %) con 100.000 simulaciones.
- **Red neuronal:** la GRU de 2 capas y 64 neuronas es la mejor. En test, su error medio en el payoff es de 0,0008 y su prima queda un 0,053 % por encima del Montecarlo de las mismas trayectorias.
- **Fórmula cerrada:** valora otro payoff (la media geométrica conjunta de la cesta) y se usa como referencia.

## Equipo

| Nombre |
|---|
| Gabriel Elorriaga |
| Alvaro Moreno |
| Ander Duñabeitia |
| Ander Sainz de la Maza |
| Iñigo Barrenechea |
| Mario Abajas |
