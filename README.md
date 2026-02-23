````markdown
# Bee-TFG — Conteo de abejas en colmenas con visión artificial (Prototipo)

Este repositorio contiene el **prototipo** de un sistema de visión artificial orientado a **detectar y trackear abejas** en la entrada de una colmena (piquera), con el objetivo final de habilitar el **conteo de entradas y salidas**.  
En esta primera fase del TFG se trabaja principalmente en **detección + tracking**, usando vídeos propios (cuando estén disponibles) y/o datasets públicos para validar el pipeline.

---

## Objetivo del proyecto (resumen)

La idea general del TFG es:

1. Colocar una **webcam** enfocando la **piquera** de una colmena.
2. Capturar vídeos cortos (p.ej. ~1 minuto).
3. Realizar anotaciones fotograma a fotograma (detección) y, si procede, generar ground truth para tracking.
4. Probar **3–4 algoritmos** (detección/tracking), entrenarlos con nuestros vídeos y **compararlos**.
5. Implementar una lógica de **conteo de entradas/salidas** (cruce de línea o región) y contrastarla con conteo manual.

> Importante: el alcance es un **prototipo**, no un sistema industrial listo para despliegue permanente en campo.

---

## Estructura del repositorio

La estructura está pensada para que el proyecto sea **ordenado, reproducible y fácil de evaluar**:

```text
bee-tfg/
├─ data/
│  ├─ raw/            # Vídeos originales (sin modificar) de la cámara / datasets
│  ├─ processed/      # Vídeos procesados (p.ej. recorte ROI, reescalado, etc.)
│  └─ annotations/    # Etiquetas/GT (YOLO/COCO/MOT) exportadas desde CVAT/Label Studio
│
├─ src/
│  └─ beecount/       # Código reutilizable (módulos del proyecto)
│
├─ scripts/           # Scripts ejecutables (pipeline: recorte, detección, tracking, exportaciones)
├─ configs/           # Configuraciones (p.ej. ROI, trackers, parámetros)
├─ runs/              # Resultados generados (vídeos con IDs, logs, métricas, etc.)
├─ notebooks/         # Exploración y prototipado rápido (Jupyter)
└─ docs/              # Material de documentación del TFG (figuras, tablas, notas, referencias)
````

```
::contentReference[oaicite:0]{index=0}
```
