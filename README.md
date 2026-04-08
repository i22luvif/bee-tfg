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

## 📂 Estructura del Repositorio

La arquitectura del proyecto sigue el estándar *Cookiecutter Data Science* para garantizar la reproducibilidad y el aislamiento de responsabilidades:

```text
bee-tfg/
├─ data/
│  ├─ raw/            # Vídeos originales (sin modificar) de la cámara / datasets
│  ├─ processed/      # Vídeos procesados (p.ej. recorte ROI, reescalado, etc.)
│  ├─ annotations/    # Etiquetas/GT (YOLO/COCO/MOT) exportadas desde CVAT/Label Studio
│  └─ images/         # Imagenes para validar
│─ datasets/
│  ├─ raw/                    <-- [Almacén de lectura] Datos originales descargados
│  │  ├─ mendeley/            <-- Secuencias originales (seq1, seq2...)
│  │  ├─ bee24/               <-- Dataset científico de validación
│  │  └─ bee2/                <-- Otro dataset científico
│  │
│  └─ ready_for_yolo/         <-- [Almacén de escritura] Datasets limpios generados
│     └─ mendeley_yolo/       <-- Resultado del script prepare_data.py
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

---

## 📹 Dataset y Vídeos de Prueba

Debido a las restricciones de tamaño de GitHub, los vídeos originales utilizados para las pruebas de este prototipo no están incluidos directamente en el repositorio. Puedes descargar los vídeos de prueba desde el siguiente enlace:

🔗 **[Descargar Dataset de Prueba (Google Drive)](https://drive.google.com/drive/folders/1UhLhz-O8WHCxz9ILM2cIvV89JZWFl5_a?usp=sharing)**

Una vez descargados, colócalos dentro de la carpeta `data/raw/` de tu proyecto local para que la aplicación web y los scripts puedan detectarlos.

---

## 💻 Instalación Local

Para reproducir este entorno en tu máquina local o servidor, sigue estos pasos:

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

---

## 🚀 Uso A: Flujo de Ejecución Core (Pipeline de Scripts)

El sistema de modelado está modularizado en 4 scripts puros de Python diseñados para ejecutarse de forma secuencial, ideal para servidores y máquinas compartidas de la Universidad.

### Fase 1: Preparación del Dataset
Extrae los datos crudos, previene la fuga de datos (*Data Leakage*) separando por secuencias de vídeo, genera anotaciones de fondo (*Negative Samples*) y construye la estructura estándar y el manifiesto `.yaml`.
```bash
python scripts/1_prepare_data.py --input datasets/raw/tu_dataset --output datasets/ready_for_yolo/dataset_formateado
```

### Fase 2: Entrenamiento del Modelo (YOLO)
Aplica *Transfer Learning* sobre el detector base. Incluye configuración automática de hardware (GPU/CPU), *Early Stopping* y técnicas de *Data Augmentation* específicas para grabaciones cenitales (rotación 180º, Mosaic, Mixup).
```bash
python scripts/2_train.py --data datasets/ready_for_yolo/dataset_formateado/data.yaml --epochs 150 --batch 8
```

### Fase 3: Inferencia y Benchmark de Trackers
Conecta el detector YOLO con 5 algoritmos de seguimiento del estado del arte (ByteTrack, BoT-SORT, OC-SORT, DeepOCSORT, StrongSORT). El código es agnóstico: acepta tanto carpetas de imágenes estandarizadas como vídeos `.mp4`.
```bash
python scripts/3_benchmark_trackers.py --model model/best_bee_medium.pt --input datasets/raw/dataset_test/test --output runs/benchmark_results
```

### Fase 4: Evaluación de Métricas (MOT Challenge)
Audita los archivos de trayectorias generados aplicando el estándar científico MOT16 con un umbral estricto de IoU (50%). Genera una tabla comparativa automática de rendimiento (MOTA, IDF1, FP, FN, etc.) lista para la memoria del proyecto.
```bash
python scripts/4_evaluate_tracking.py --gt datasets/raw/dataset_test/test --benchmark_dir runs/benchmark_results
```

---

## 🌐 Uso B: Interfaz de Usuario Web (Streamlit)

Si deseas utilizar la interfaz visual interactiva para cargar vídeos y probar el modelo entrenado de forma amigable:

Asegúrate de colocar al menos un vídeo de prueba en la carpeta `data/raw/` y tu modelo entrenado en la carpeta `model/`, y ejecuta:

```bash
streamlit run app.py
```