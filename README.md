# Hybrid Waste Classifier

Clasificador de residuos por imagen que combina **procesamiento clasico de imagenes + SVM**
con un **verificador CNN (MobileNetV2)** que solo se activa cuando el SVM tiene baja confianza.
Incluye un dashboard en Streamlit para clasificar imagenes, ver el preprocesamiento paso a paso
y obtener a que tacho va cada residuo.

El objetivo es asistir la **segregacion de residuos en el punto de origen**: un solo residuo mal
puesto puede contaminar un lote entero de reciclables, y muchas veces la persona no sabe en que
contenedor va el objeto. La herramienta sugiere el tacho correcto con datos del material.

---

## Clases

El proyecto parte de 12 clases originales que se fusionan en **9 clases finales**:

`battery` · `cardboard` · `glass` · `metal` · `organic` · `paper` · `plastic` · `textile` · `trash`

La fusion principal agrupa ropa y calzado en `textile`. El split de entrenamiento se balancea a
750 imagenes por clase (submuestreo en clases dominantes, data augmentation en las demas), mientras
que validacion y test se mantienen con datos reales sin modificar para una evaluacion honesta.

---

## Motivacion del enfoque hibrido

El analisis exploratorio mostro que el **color RGB por si solo no separa bien las 9 clases**:
varios grupos (vidrio, pila, papel, plastico) comparten histogramas dominados por fondos claros,
y clases como textil tienen distribuciones demasiado variables. Por eso el sistema no se queda en
color: agrega descriptores de **textura** (LBP, GLCM, Gabor) y **forma**, y para los casos dudosos
delega a una **CNN** que aprende representaciones mas ricas.

La logica del hibrido:

- El **SVM clasico** es rapido y barato; resuelve los casos faciles con features hechas a mano.
- Cuando la confianza del SVM cae por debajo de un umbral `theta`, recien entra la **CNN**,
  mas precisa pero mas costosa.

Esto ahorra computo, lo que importa si se busca correr en dispositivos de bajo recurso (edge).

---

## Pipeline

1. **Segmentacion del objeto** (`src/segmentation/`)
   - Estimacion de densidad de bordes sobre el canal de saturacion (S).
   - Filtro bilateral pasabajo condicional para imagenes con mucha textura de fondo.
   - Bordes Sobel sobre los canales S y V.
   - Operaciones morfologicas (apertura, cierre, relleno de contornos) para obtener la mascara.

2. **Extraccion de features** (sobre el objeto y la imagen completa)
   - Textura: histograma LBP uniforme y propiedades GLCM (contrast, homogeneity, energy, correlation).
   - Gabor: media y desviacion en 3 frecuencias x 4 orientaciones.
   - Color: estadisticos HSV y LAB (media, desviacion, mediana, percentiles).
   - Forma: area, perimetro, extent, solidity, aspect ratio, circularidad y momentos de Hu.
   - Intensidad: media, desviacion, asimetria, curtosis, entropia.

3. **Clasificacion** en tres modos seleccionables
   - **SVM solo**
   - **Hibrido SVM + CNN** (verificacion por umbral de confianza)
   - **CNN sola**

4. **Mapeo a tacho** segun esquema de colores (NTP 900.058) con datos del material.

---

## Estructura del repositorio

```
.
├── app.py                  # dashboard Streamlit (modos, preprocesamiento, simulacion de tacho)
├── notebooks/
│   ├── 01_exploracion_dataset...        # distribucion de clases, resoluciones, histogramas de color
│   ├── 02_segmentacion_hsv...           # ajuste y validacion visual de la segmentacion
│   ├── 03_extraccion_caracteristicas... # features de train/val/test a CSV
│   ├── 04_analisis_separabilidad...     # PCA/t-SNE, baselines, importancia de features
│   ├── 05_entrenamiento_ablation...     # comparacion de modelos, ablation, GridSearch SVM
│   ├── 06_pipeline_hibrido...           # entrenamiento CNN y enrutamiento por confianza
│   └── 07_evaluacion_final...           # F1 por clase, activacion CNN, analisis de fallos
├── src/
│   └── segmentation/
│       ├── edge_detection.py
│       └── morphological_ops.py
├── results/
│   ├── models/             # svm_clasico.pkl, scaler_clasico.pkl, features_clasico.pkl, cnn_verificador.h5
│   ├── features/           # CSV de features por split
│   └── figures/            # figuras generadas (ignoradas por git)
├── data/                   # raw / processed / segmented (ignoradas por git)
├── requirements.txt
└── .gitignore
```

---

## Flujo de notebooks

| Notebook | Que hace |
|----------|----------|
| 01 | Explora el dataset: distribucion de clases (original vs fusionada), resoluciones e histogramas de color RGB. Justifica usar textura y forma ademas de color. |
| 02 | Ajusta y valida visualmente la segmentacion HSV + Sobel + morfologia clase por clase, y procesa todos los splits guardando mascaras y objetos. |
| 03 | Extrae todas las features sobre objeto e imagen completa para train, val y test, y las guarda en CSV. |
| 04 | Analiza separabilidad con PCA y t-SNE, entrena baselines (RandomForest, SVM), genera matrices de confusion e importancia de features. |
| 05 | Compara seis clasificadores, hace ablation por familia de features, ajusta hiperparametros del SVM con GridSearch y evalua en test. Guarda el modelo clasico. |
| 06 | Entrena el verificador CNN (MobileNetV2 transfer learning) y construye el pipeline hibrido por umbral de confianza. |
| 07 | Evaluacion final: F1 por clase (SVM vs hibrido), tasa de activacion de la CNN, analisis de errores y sensibilidad al umbral. |

---

## Uso

```bash
pip install -r requirements.txt
streamlit run app.py
```

En el dashboard:

- Sube **una o varias imagenes**.
- Elige el **modo** en la barra lateral (SVM solo / Hibrido / CNN sola).
- Ajusta el **umbral `theta`** (solo en modo hibrido).
- Activa **Mostrar preprocesamiento** para ver el canal S, bordes Sobel, morfologia, mascara,
  objeto segmentado y overlay, con sus metricas.

Por cada imagen obtienes la clase, el tacho sugerido con su color, una simulacion del residuo
cayendo en el contenedor, datos del material y la distribucion de probabilidad.

---

## Requisitos

opencv-python, numpy, scikit-learn, scikit-image, scipy, pandas, joblib, matplotlib, seaborn,
tensorflow, streamlit, Pillow, tqdm (ver `requirements.txt`).

Los modelos entrenados en `results/models/` son necesarios para correr el app al clonar.
Si `cnn_verificador.h5` supera los 100 MB, usa Git LFS para versionarlo.

---

## Limitaciones

- Varias clases se confunden por color o textura (vidrio, pila, papel, plastico con fondos claros).
- El split de entrenamiento esta balanceado artificialmente, asi que las metricas internas no
  reflejan la distribucion real de residuos.
- La herramienta funciona mejor como **asistente que reduce el error humano**, no como decisor autonomo.
- Los tiempos de degradacion mostrados son cifras divulgativas de referencia; para uso formal
  conviene respaldarlas con una fuente oficial.

---

