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

**4. Inicia la aplicación:**
```bash
streamlit run app.py
```
*(Asegúrate de colocar al menos un vídeo de prueba en la carpeta `data/raw/` y tu modelo entrenado en `model/`).*

## 🛤️ Siguientes Pasos (Fase 2)

- Obtener vídeos reales de la piquera de la colmena objetivo.
- Anotar manualmente un subconjunto de los vídeos reales para hacer *Fine-tuning* del modelo actual.
- Implementar la Lógica de Cruce de Línea en `app.py`: Trazar una línea virtual en la pantalla y calcular vectores de movimiento de los IDs para diferenciar qué abejas entran y cuáles salen.
- Extracción de métricas de rendimiento y redacción de la memoria del TFG.

