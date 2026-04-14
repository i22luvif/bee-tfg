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
├─ data/                  # Datos brutos del usuario
│  ├─ raw/                # Vídeos originales (sin modificar) de la cámara
│  ├─ processed/          # Vídeos procesados (recortes, reescalados, etc.)
│  └─ annotations/        # Etiquetas generadas y exportadas
├─ datasets/              # [Ignorado en Git] Almacén de Datasets para IA
│  ├─ raw/                # Datasets científicos descargados (BEE24, Mendeley...)
│  └─ ready_for_yolo/     # Datasets limpios generados por el script 1 (preparados para YOLO)
├─ model/                 # Modelos definitivos
│  └─ best_bee.pt         # Modelo YOLO final entrenado con nuestros datos
├─ runs/                  # [Ignorado en Git] Resultados generados
│  └─ benchmark_results/  # Archivos .txt (MOT16) generados por los trackers
├─ scripts/               # Core de ejecución y lógica del TFG
│  ├─ 0_pre_annotate.py         # Motor de auto-etiquetado (Frame Skipping)
│  ├─ 1_prepare_data.py         # Formateo de datos y prevención de Data Leakage
│  ├─ 2_train.py                # Entrenamiento YOLO asíncrono
│  ├─ 3_benchmark_trackers.py   # Inferencia multiobjeto (ByteTrack, BoT-SORT...)
│  ├─ 4_evaluate_tracking.py    # Evaluación matemática (MOTMetrics)
│  └─ app.py                    # Aplicación web principal (Dashboard Streamlit)
├─ .gitignore             # Reglas de exclusión de archivos pesados
├─ osnet_x0_25_msmt17.pt  # Modelo de Re-Identificación (usado por los Trackers)
├─ README.md              # Documentación principal
├─ requirements.txt       # Dependencias estrictas del entorno virtual
└─ yolo26m.pt             # Modelo YOLO base (pesos pre-entrenados)
````

-----

## 📹 Dataset y Vídeos de Prueba

Debido a las restricciones de tamaño de GitHub, los vídeos originales utilizados para las pruebas de este prototipo no están incluidos directamente en el repositorio. Puedes descargar los vídeos de prueba desde el siguiente enlace:

🔗 **[Descargar Dataset de Prueba (Google Drive)](https://www.google.com/search?q=%23)** *(Recuerda poner tu enlace real aquí)*

Una vez descargados, colócalos dentro de la carpeta `data/raw/` de tu proyecto local para que la aplicación web y los scripts puedan detectarlos.

-----

## 💻 Instalación Local

Para reproducir este entorno en tu máquina local o servidor, sigue estos pasos:

**1. Clona el repositorio y crea el entorno virtual:**

```bash
git clone [https://github.com/TU_USUARIO/bee-tfg.git](https://github.com/TU_USUARIO/bee-tfg.git)
cd bee-tfg
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/Mac:
source .venv/bin/activate
```

**2. Instalación del Motor de IA (PyTorch):**
Debido a las dependencias de hardware (NVIDIA, AMD, Apple), debes instalar la versión de PyTorch que corresponda a tu equipo ANTES de instalar el resto del proyecto.

  * **Para NVIDIA (CUDA) - Windows/Linux:**

<!-- end list -->

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

  * **Para AMD (ROCm) - Linux:**

<!-- end list -->

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/rocm5.6](https://download.pytorch.org/whl/rocm5.6)
```

  * **Para Apple Silicon (M1/M2/M3) o CPU genérica:**

<!-- end list -->

```bash
pip install torch torchvision
```

**3. Instalación de las dependencias del proyecto:**
Una vez instalado PyTorch, instala el resto de librerías estándar:

```bash
pip install -r requirements.txt
```

-----

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
python scripts/3_benchmark_trackers.py --model model/best_bee.pt --input datasets/raw/dataset_test/test --output runs/benchmark_results
```

### Fase 4: Evaluación de Métricas (MOT Challenge)

Audita los archivos de trayectorias generados aplicando el estándar científico MOT16 con un umbral estricto de IoU (50%). Genera una tabla comparativa automática de rendimiento (MOTA, IDF1, FP, FN, etc.) lista para la memoria del proyecto.

```bash
python scripts/4_evaluate_tracking.py --gt datasets/raw/dataset_test/test --benchmark_dir runs/benchmark_results
```

-----

## 🌐 Uso B: Interfaz Web (Panel de Control Streamlit)

Si prefieres gestionar todo el flujo de trabajo de forma gráfica en lugar de utilizar comandos individuales en la terminal, el proyecto cuenta con un **Panel de Control unificado**. Desde él podrás preparar datasets, lanzar entrenamientos en segundo plano, evaluar métricas MOT y auto-etiquetar vídeos nuevos con un solo clic.

Asegúrate de colocar al menos un vídeo de prueba en la carpeta `data/raw/` y tu modelo base/entrenado en la carpeta `model/`. Para iniciar la aplicación, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
streamlit run scripts/app.py
```