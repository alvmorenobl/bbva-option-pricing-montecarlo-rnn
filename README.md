# Reto_09_Grupo_Verde_Claro

Cosas a tener en cuenta:
- Los scripts **sí deben ejecutarse en el orden propuesto en este README** (a diferencia de otros retos donde son independientes), ya que cada uno consume la salida del anterior: unificación de datos → EDA → selección de cartera → simulación Montecarlo → cálculo de payoff → comparación con la RNN.
- Se necesitan los CSVs de las 26 acciones del Dow Jones facilitados por los PMs, así como conexión a internet para completar datos vía `yfinance` (Yahoo Finance).
- La carpeta de datos **no se sube a GitHub** (ni originales ni generados), solo a la carpeta de Drive del equipo (`R9_Entrega_EQUIPO`), tal y como exige la normativa del reto.

## Datos

En estas carpetas están o se tienen que ubicar todos los datos utilizados.

### Originales

Aquí se tienen que incluir los CSVs originales otorgados al principio del reto:

* 26 archivos CSV, uno por cada acción del Dow Jones (columnas `open`, `high`, `low`, `last`, `close`, `volume`, `date`, `symbol`, `exchange`).

### Transformados

Aquí se ubican los datos generados durante el proceso:

* `dow_jones_clean.csv`: las 26 acciones unificadas, limpias y completadas con datos de Yahoo Finance.
* `cartera_3_activos.csv`: serie histórica únicamente de las 3 acciones finalmente seleccionadas para la cartera.
* `X_simulaciones.npy`: matriz de precios simulados por Montecarlo, de dimensiones (n_simulaciones, n_pasos, n_activos).
* `Y_payoff.npy`: matriz de payoffs calculados a partir de `X_simulaciones.npy`, usada como target de la RNN.

## Preprocesamiento

En este script (`Preprocesamiento.py`) se realiza:

* La lectura de los 26 CSV individuales de las acciones del Dow Jones.
* La unificación de todos en un único DataFrame.
* La limpieza de nulos, duplicados y formatos de fecha.
* La descarga y cruce con datos complementarios de `yfinance` para completar el conjunto (recordemos que solo se facilitan 26 de las 30 empresas).
* La exportación del resultado a `dow_jones_clean.csv`.

## EDA

En este script (`EDA.py`) se hace el análisis exploratorio sobre `dow_jones_clean.csv`:

* Comprobación de la frecuencia de los datos (diaria vs intradía) y agregación si procede.
* Resumen de calidad: nulos, duplicados, rango de fechas, registros por empresa.
* Estadística descriptiva del precio de cierre por empresa.
* Evolución temporal de precios normalizados (base 100).
* Cálculo de retornos logarítmicos diarios y ajuste a distribución normal.
* Ranking de volatilidad anualizada por acción.
* Matriz de correlación entre las 26 acciones (se reutiliza después en `Montecarlo.py` para la descomposición de Cholesky).
* Análisis de volumen/liquidez.

## Selección de cartera

En `Seleccion_cartera.py` se decide, a partir de los resultados del EDA, qué 3 acciones forman la cartera subyacente de la opción:

* Aplicación de los criterios cuantitativos (volatilidad, correlación, ratio de Sharpe) y cualitativos (sector, modelo de negocio) definidos por el equipo.
* Cálculo de los pesos óptimos de la cartera (mínima varianza / máximo Sharpe, según la estrategia elegida).
* Generación de la frontera eficiente para las 3 acciones seleccionadas.
* Exportación de la serie histórica de las 3 acciones a `cartera_3_activos.csv`.

## Series temporales

En `Series_temporales.py` se hace la descomposición (tendencia, estacionalidad, residuo) de cada una de las 3 series temporales de `cartera_3_activos.csv`, para entender su comportamiento antes de simular su evolución futura.

## Montecarlo

En `Montecarlo.py` se genera el conjunto de datos sintéticos:

* Cálculo de $S_0$, $\mu$ (tasa libre de riesgo), $\sigma$ (vector de volatilidades) y $\rho$ (matriz de correlaciones) a partir de la cartera seleccionada.
* Descomposición de Cholesky de la matriz de correlaciones para introducir dependencia entre los movimientos brownianos simulados.
* Simulación mediante Movimiento Browniano Geométrico de la evolución de los 3 activos a 5 años (20 pasos temporales, mínimo 1000 simulaciones).
* Exportación de la matriz resultante a `X_simulaciones.npy`.

## Payoff

En `Payoff.py` se calcula, a partir de `X_simulaciones.npy`:

* El payoff de la opción asiática geométrica (media geométrica de las 3 series simuladas menos el strike, con el `[...]+`).
* El descuento a valor presente (prima) para distintos instantes del intervalo de 5 años.
* Exportación de los payoffs a `Y_payoff.npy`.

## FormulaCerradaBBVA

En `FormulaCerradaBBVA.py` se implementa la fórmula analítica cerrada facilitada/investigada para la valoración de la call asiática geométrica, usada como método de referencia adicional frente a Montecarlo.

## RNN

En `RNN.py` se entrena el modelo basado en datos:

* Preparación de `X_simulaciones.npy` (input) e `Y_payoff.npy` (target) en conjuntos de train/test.
* Definición y entrenamiento de distintas arquitecturas recurrentes (RNN simple, LSTM, GRU, Bi-LSTM) para comparar cuál predice mejor el payoff.
* Selección de la arquitectura final según error de predicción.

## Comparacion_metodos

En `Comparacion_metodos.py` se comparan los resultados de:

* La media geométrica + Montecarlo.
* La fórmula cerrada de BBVA.
* La RNN entrenada.

Se generan la tabla de predicción del valor de la opción en el momento *t* y los criterios de evaluación del error (validación cruzada incluida), tal y como exige el entregable.

## Visualización

Carpeta `viz/` con los gráficos interactivos en HTML/JS del proyecto (`index.html`, `charts.js`), usados en el informe y en la presentación.

## Gestión del proyecto

Carpeta `docs/gestion_proyecto/` con las evidencias del proceso de gestión (actas, desglose de trabajo, políticas de equipo definidas en el Hito 0).

## Equipo

| Nombre |
|---|
| Gabriel Elorriaga |
| Alvaro Moreno |
| Ander Duñabeitia |
| Ander Sainz de la Maza |
| Iñigo Barrenechea |
| Mario Abajas |

**Repositorio:** https://github.com/mario-cr7/reto09-verde_claro
