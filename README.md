# 🐝 Bee-TFG — Conteo de abejas en colmenas con visión artificial

Este repositorio contiene el prototipo y código fuente de un Trabajo de Fin de Grado (TFG) orientado a la **detección y rastreo (tracking) de abejas** en la entrada de una colmena (piquera). El objetivo final es habilitar un sistema automático de **conteo de entradas y salidas** utilizando técnicas avanzadas de Visión por Computador e Inteligencia Artificial.

---

## 📌 Objetivo del Proyecto

La idea general del TFG abarca las siguientes fases:
1. Colocar una **webcam** enfocando la piquera de una colmena.
2. Capturar vídeos de muestra y construir un dataset propio.
3. Entrenar y evaluar modelos de detección de objetos (YOLO).
4. Probar y comparar algoritmos de rastreo multiobjeto (MOT) como ByteTrack, BoT-SORT...
5. Implementar una lógica geométrica de **conteo de entradas/salidas** (cruce de línea o región) contrastada con conteo manual.

> **Importante:** El alcance de este proyecto es el desarrollo de un **prototipo funcional** para validar el pipeline de software, no un sistema industrial listo para despliegue permanente a la intemperie.

---

## 🚀 Estado Actual: Fase 1 Completada (Prototipo Interactivo)

Actualmente, el proyecto cuenta con un **pipeline end-to-end validado** mediante una aplicación web interactiva. En lugar de usar únicamente scripts por consola, se ha desarrollado una interfaz gráfica que permite procesar vídeos en tiempo real.

### Hitos logrados:
* **Entrenamiento del Modelo:** Se ha entrenado un modelo base (`yolo26n.pt`) durante 50 épocas en la nube (Google Colab) utilizando un dataset público de imágenes de abejas, obteniendo los pesos finales (`best_bee.pt`).
* **Desarrollo del Frontend:** Creación de un panel de control local utilizando **Streamlit**.
* **Integración del Tracker:** Implementación de persistencia temporal acoplando algoritmos MOT (BoT-SORT y ByteTrack) al motor de inferencia.
* **Conteo de Individuos Únicos:** Implementación de lógica matemática mediante conjuntos (`sets`) para almacenar los IDs temporales y contabilizar el número total de abejas únicas que aparecen en el vídeo, evitando el conteo redundante.

### 🛠️ Características del Panel de Control (App)
La interfaz actual permite al investigador modificar hiperparámetros en caliente:
- Selección del modelo de IA (`.pt`).
- Ajuste de **Confianza mínima (conf)** y **Solapamiento máximo (IoU)**.
- Selección de la resolución de análisis (`imgsz`).
- Intercambio en tiempo real entre algoritmos de rastreo (`botsort.yaml` vs `bytetrack.yaml`).
- Visualización en directo del procesado fotograma a fotograma (OpenCV).
- Exportación y descarga directa del vídeo resultante en formato `.mp4`.

---

## 📂 Estructura del Repositorio

La arquitectura del proyecto sigue el estándar *Cookiecutter Data Science* para garantizar la reproducibilidad y el aislamiento de responsabilidades:

``` text
bee-tfg/
├─ data/
│  ├─ raw/            # Vídeos originales (sin modificar) de la cámara / datasets
│  ├─ processed/      # Vídeos procesados (p.ej. recorte ROI, reescalado, etc.)
│  ├─ annotations/    # Etiquetas/GT (YOLO/COCO/MOT) exportadas desde CVAT/Label Studio
│  └─ images/         # Imagenes para validar
│─ datasets/
│  ├─ raw/                     <-- [Almacén de lectura] Datos originales descargados
│  │  ├─ mendeley/             <-- Tus secuencias originales (seq1, seq2...)
│  │  ├─ bee24/                <-- El dataset científico de validación
│  │  └─ bee2/                 <-- El otro dataset científico
│  │
│  └─ ready_for_yolo/          <-- [Almacén de escritura] Datasets limpios generados por ti
│     └─ mendeley_yolo/        <-- ¡AQUÍ irá el resultado del script prepare_data.py!
│        ├─ train/             
│        ├─ val/               
│        ├─ test/              
│        └─ data.yaml
├─ src/               # Código reutilizable (módulos del proyecto)
├─ scripts/           # Scripts ejecutables (pipeline: recorte, detección, exportaciones)
├─ configs/           # Configuraciones (p.ej. ROI, trackers, parámetros)
├─ runs/              # Resultados generados (vídeos con IDs, logs, métricas, etc.)
├─ notebooks/         # Exploración y prototipado rápido (Jupyter)
├─ model/             # Modelos entrenados (ej. best_bee.pt)
├─ docs/              # Material de documentación del TFG (figuras, tablas, notas)
├─ app.py             # Aplicación principal (Interfaz web con Streamlit)
└─ requirements.txt   # Dependencias del proyecto
```

## 💻 Instalación y Uso local

Para reproducir este entorno en tu máquina local, sigue estos pasos:

**1. Clona el repositorio:**
```bash
git clone [https://github.com/TU_USUARIO/bee-tfg.git](https://github.com/TU_USUARIO/bee-tfg.git)
cd bee-tfg
```

**2. Crea y activa un entorno virtual:**
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

**3. Instala las dependencias:**
```bash
pip install -r requirements.txt
```

**4. Introduce los videos y el modelo dentro de sus respectivas carpetas**
*(Asegúrate de colocar al menos un vídeo de prueba en la carpeta `data/raw/` y tu modelo entrenado en `model/`).*


**5. Inicia la aplicación:**
```bash
streamlit run app.py
```


## 📹 Dataset y Vídeos de Prueba
Debido a las restricciones de tamaño de GitHub, los vídeos originales utilizados para las pruebas de este prototipo no están incluidos directamente en el repositorio.

Puedes descargar los vídeos de prueba desde el siguiente enlace:

https://drive.google.com/drive/folders/1UhLhz-O8WHCxz9ILM2cIvV89JZWFl5_a?usp=sharing

Una vez descargados, colócalos dentro de la carpeta data/raw/ de tu proyecto local para que la aplicación web pueda detectarlos.


## 🛤️ Siguientes Pasos (Fase 2)

- Obtener vídeos reales de la piquera de la colmena objetivo.
- Anotar manualmente un subconjunto de los vídeos reales para hacer *Fine-tuning* del modelo actual.
- Implementar la Lógica de Cruce de Línea en `app.py`: Trazar una línea virtual en la pantalla y calcular vectores de movimiento de los IDs para diferenciar qué abejas entran y cuáles salen.
- Extracción de métricas de rendimiento y redacción de la memoria del TFG.


## 🚀 Fase 2 Seguimiento Avanzado (Multi-Object Tracking - MOT)

Para dotar al Trabajo de Fin de Grado (TFG) del máximo rigor científico, se ha trascendido el uso de los rastreadores integrados por defecto en YOLO (que limitaban la comparativa) y se ha implementado la librería de investigación académica **BoxMOT**. Esto ha permitido realizar un *benchmark* (evaluación comparativa) enfrentando a 5 de los algoritmos de seguimiento multiobjeto más punteros del estado del arte.


### 🎯 Los 5 Algoritmos Evaluados (El "Big 5")

1. **ByteTrack**: Algoritmo puramente cinemático/geométrico. Destaca por su alta velocidad, ya que asocia objetos basándose exclusivamente en el solapamiento de las cajas delimitadoras (IoU) sin requerir redes neuronales secundarias complejas.
2. **BoT-SORT**: Mejora los algoritmos tradicionales añadiendo compensación del movimiento de la cámara y un mejor manejo del estado de los objetos. En nuestras pruebas iniciales, ha demostrado ser el rastreador **más estable** para lidiar con las oclusiones masivas en la entrada de la colmena (piquera).
3. **OC-SORT (Observation-Centric SORT)**: Diseñado específicamente para mantener el rastro de objetos que se ocultan temporalmente y que presentan movimientos caóticos o no lineales, características idóneas para el vuelo de los insectos.
4. **StrongSORT**: Evolución avanzada que utiliza ReID (Re-Identificación) mediante una red neuronal secundaria (ej. *osnet*) para extraer características visuales de los objetos y diferenciarlos por su "textura".
5. **DeepOCSORT**: La evolución de vanguardia del clásico DeepSORT, combinando la robustez geométrica de OC-SORT con la extracción de características profundas.
   > **⚠️ Nota Técnica de Actualización (DeepSORT vs DeepOCSORT):** > Durante el desarrollo de la investigación, se constató que el algoritmo clásico `DeepSORT` ha quedado obsoleto y ha sido eliminado de las versiones recientes de librerías modernas de tracking (como `boxmot`), en parte debido a incompatibilidades con arquitecturas matemáticas modernas (NumPy 2.0). Por rigor académico, fue sustituido en la comparativa por su variante moderna y superior, **DeepOCSORT**.

### 🧠 Descubrimiento Científico Clave (ReID vs Geometría)
Las pruebas empíricas han revelado una conclusión vital para el estudio de insectos: **Los algoritmos basados en el reconocimiento de apariencia (ReID)**, como StrongSORT o DeepOCSORT, resultan contraproducentes para el rastreo de abejas. 

Al ser estos insectos visualmente idénticos entre sí, la red neuronal secundaria (*feature extractor*) se confunde al intentar encontrar diferencias en sus texturas, lo que provoca constantes cambios de identificador (Fragmentación de IDs). Por el contrario, los **trackers puramente geométricos o cinemáticos** (BoT-SORT, ByteTrack, OC-SORT) ofrecen un rendimiento muy superior al guiarse únicamente por trayectorias espaciales.

### 🛠️ Próximos Pasos
* Integración del modelo entrenado de mayor profundidad (**YOLOv8 Medium**) para reducir los falsos negativos temporales (parpadeos de detección) que alimentan al Tracker.
* Implementación de una interfaz gráfica avanzada en **Streamlit** para visualizar las métricas y los seguimientos en tiempo real.

