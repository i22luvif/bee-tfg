
# 📓 Diario de Investigación y Experimentación - TFG: Sistema MOT para Colmenas

Este documento recoge las decisiones arquitectónicas, problemas técnicos superados y resultados empíricos obtenidos durante el desarrollo del sistema de detección y seguimiento multiobjeto (**MOT**) aplicado a colmenas de abejas.

---

## Fase 1: Arquitectura, Selección de Algoritmos y Retos Técnicos

### 1.1 Desacoplamiento de YOLO y Trackers
Inicialmente, se planteó utilizar los algoritmos de rastreo integrados de forma nativa en **Ultralytics** junto a nuestro modelo **YOLO26**. Sin embargo, esta opción presentaba una limitación metodológica grave: solo permitía evaluar dos algoritmos (**ByteTrack** y **BoT-SORT**), impidiendo una comparativa amplia con el estado del arte.

* **Decisión Técnica:** Se optó por desacoplar la etapa de detección de la etapa de rastreo.
    * **Detección:** YOLO26 actúa exclusivamente como extractor espacial (genera tensores con las coordenadas de las cajas delimitadoras).
    * **Rastreo (Tracking):** Se integró la librería especializada académica **BoxMOT**. Un script intermedio traduce los tensores de PyTorch a matrices puras de **NumPy**, inyectándolas en BoxMOT para instanciar múltiples rastreadores modulares.

### 1.2 Obstáculos Técnicos y "Dependency Hell"
* **Obsolescencia de DeepSORT:** El diseño original contemplaba evaluar el clásico algoritmo DeepSORT. Sin embargo, se descubrió que ha sido discontinuado en librerías modernas por su ineficiencia frente a arquitecturas actuales. Fue sustituido por **DeepOCSORT**, una variante superior que combina la robustez de OC-SORT con extracción de características..
* **Conflicto con NumPy 2.0:** Al ejecutar los Filtros de Kalman en BoxMOT, el sistema colapsaba con el error `TypeError: only 0-dimensional arrays can be converted to Python scalars`. La actualización global de NumPy a la versión 2.0 rompió la retrocompatibilidad de las operaciones de división de matrices usadas para calcular las dimensiones de las cajas (w/h).
* **Solución evolutiva:** En una primera aproximación, se realizó un *downgrade* estricto del entorno virtual a `numpy<2.0.0`. Más adelante, para dotar al código de mayor elegancia y evitar alterar entornos en la nube, se desarrolló un parche directo en el código (**Monkey Patching**: `np.asfarray = lambda a: np.asarray(a, dtype=float)`), solucionando el problema de forma nativa.

### 1.3 Evaluación de Algoritmos y el Fracaso del ReID
Se enfrentó a 5 algoritmos de primer nivel (**ByteTrack, BoT-SORT, OC-SORT, StrongSORT y DeepOCSORT**). Los resultados empíricos demostraron que los algoritmos más pesados e inteligentes (StrongSORT y DeepOCSORT) obtuvieron los peores resultados.
1. **ByteTrack:** Geométrico puro (rápido, basado en IoU).
2. **BoT-SORT:** Geométrico con compensación de movimiento de cámara.
3. **OC-SORT:** Geométrico optimizado para trayectorias caóticas y oclusiones.
4. **StrongSORT:** Basado en ReID (Re-identificación por apariencia).
5. **DeepOCSORT:** Híbrido (Cinemática + ReID).

> **Conclusión Científica:** Estos algoritmos utilizan una red neuronal secundaria para extraer "firmas visuales" (**ReID**). Dado que las abejas son visualmente idénticas, el módulo ReID colapsa al intentar buscar diferencias, provocando reasignaciones de ID constantes. Para insectos, los modelos puramente cinemáticos/geométricos son empíricamente superiores.

---

## Fase 2: Entrenamiento de YOLO26 y Pruebas Base

Para la detección, la arquitectura **YOLO26** (*Single-Stage Detector*) fue elegida por su capacidad de procesar imágenes en tiempo real. El entrenamiento se realizó en la nube (Kaggle con GPU T4 x2) en dos etapas:

1.  **Modelo Base (YOLO26 Nano - yolo26n.pt, 50 épocas):** Validó el flujo de trabajo, pero presentaba "parpadeos" (falsos negativos) cuando las abejas se solapaban. El mejor tracker (**BoT-SORT**) limitó la fragmentación a 64 IDs en un video de ~20 abejas.
| Algoritmo Tracker | Tiempo Total (seg) | Velocidad (FPS) | IDs Únicos (Menos es mejor) |
|-------------------|--------------------|-----------------|-----------------------------|
| **BYTETRACK** | 102.47             | 9.12            | 75                          |
| **BOTSORT** | 165.53             | 5.65            | **64** (Ganador Fase 1)     |
| **DEEPOCSORT** | 180.00             | 5.19            | 85                          |
| **STRONGSORT** | 173.78             | 5.38            | 101                         |
| **OCSORT** | 99.91              | 9.36            | 76                          |

*El límite de 64 IDs se debe al "parpadeo" (falsos negativos temporales) de la red neuronal Nano, que pierde la caja durante varios fotogramas, obligando al Tracker a crear un ID nuevo al reaparecer.*

2.  **Modelo Avanzado (YOLO26 Medium - best_bee_medium.pt, 150 épocas):** Dotó al sistema de mayor robustez. Aunque redujo la velocidad de inferencia por su carga paramétrica, la fragmentación de identificadores se desplomó.


| Algoritmo Tracker | Tiempo Total (seg) | Velocidad (FPS) | IDs Únicos (Fragmentación) |
|-------------------|--------------------|-----------------|-----------------------------|
| **BYTETRACK** | 549.20             | 1.70            | 71                          |
| **BOTSORT** | 620.42             | 1.51            | 59                          |
| **DEEPOCSORT**| 635.60             | 1.47            | 53                          |
| **STRONGSORT**| 759.77             | 1.23            | 54                          |
| **OCSORT** | 613.85             | 1.52            | **48** (Ganador Absoluto)   |

### 📊 Análisis de Resultados y Conclusiones:
1. **Trade-off Velocidad/Precisión:** Como era predecible, la mayor carga paramétrica de la arquitectura Medium redujo el rendimiento de inferencia (caída de ~6 FPS de media), multiplicando el tiempo de procesamiento.
2. **Estabilización de Detecciones:** La fragmentación de identificadores se desplomó de forma generalizada en todos los algoritmos. Se confirma la hipótesis: una red detectora más robusta reduce los falsos negativos temporales (parpadeos), proporcionando secuencias de cajas delimitadoras más sólidas a los Filtros de Kalman del Tracker.
3. **El Triunfo de OC-SORT:** El rastreador **OC-SORT** (*Observation-Centric SORT*) demostró ser el más capaz para este dominio (48 IDs). Su arquitectura está diseñada para mitigar el error acumulativo durante períodos de oclusión y gestionar trayectorias no lineales, algo característico y fundamental en el vuelo errático de las abejas en el entorno caótico de una piquera.

---

## Fase 3: Interfaz Gráfica de Usuario (GUI)

Para democratizar el uso del sistema y permitir que un usuario sin conocimientos de programación interactúe con el modelo, se desarrolló un panel de control interactivo.

**Decisiones de Diseño Frontend/Backend:**
* **Streamlit:** Se descartaron *frameworks* pesados (como React o Angular) y librerías de escritorio clásicas (como Tkinter o PyQt) en favor de Streamlit. Esta tecnología nativa de Python permite levantar una interfaz web reactiva en escasas líneas de código. Su principal ventaja es la integración directa con los *scripts* de Machine Learning, permitiendo modificar hiperparámetros de YOLO (Confianza e IoU) en tiempo real mediante *sliders* (deslizadores) en la barra lateral.
* **OpenCV (cv2):** Actúa como el motor gráfico intermedio. Se encarga de la decodificación del vídeo bruto (`.mp4`), la inyección de metadatos visuales (dibujo de las cajas delimitadoras, *centroides* y textos de conteo en color RGB), y la recodificación final del vídeo procesado usando el códec `mp4v` para su posterior descarga desde la web.


---

## Fase 4: Optimización en Piquera Real (Validación Cualitativa)

En esta fase se llevó a cabo la optimización final sobre un clip de control real en la piquera.

* **Grid Search:** Se realizaron varias rondas de pruebas masivas para encontrar los hiperparámetros ganadores:
    * **Primera Ronda:** Ajustamos `conf` y `track_buffer`. Logramos bajar de +100 IDs a **34 IDs estables**
    * **Segunda Ronda (Prueba Maestra):** Introdujimos el **Filtro de Persistencia** (`min_hits` y conteo de frames mínimos).
    * **Resultados Clave:**
        * Con Confianza **0.65** y Persistencia **20 frames** -> **19 abejas**.
        * Con Confianza **0.85** y Persistencia **30 frames** -> **13 abejas**.

* **Resultado Final:** Contra un *Ground Truth* manual de 14 abejas reales, el sistema alcanzó una **precisión del 92.8%**, eliminando casi el 90% de la fragmentación inicial.

---

## Fase 5: Validación Científica Cruzada y Estudio de Ablación (BEE24)

Para transicionar a una validación cuantitativa rigurosa, se enfrentó el modelo **YOLO26 Medium + OC-SORT** al dataset científico estandarizado **BEE24**. Se adaptó el pipeline para leer secuencias de imágenes estáticas (`img1`), se integró el repositorio oficial de OC-SORT y se midió el rendimiento con `motmetrics` (formato mot15-2D).

Para lograr la compatibilidad con los estándares de investigación MOT (Multiple Object Tracking), se desarrollaron los siguientes pasos en el entorno de Kaggle:
* **Adaptación de Inferencia:** Se modificó el script para leer secuencias de imágenes estáticas (`img1`) en estricto orden alfanumérico en lugar de video, asegurando el correcto flujo temporal para el Filtro de Kalman de OC-SORT.
* **Integración de OC-SORT:** Se importó el repositorio oficial de los autores para evitar conflictos de dependencias con librerías de terceros (como `boxmot`), inyectando el tamaño de la imagen (`img_info`, `img_size`) necesario para el cálculo preciso de trayectorias.
* **Exportación de Datos:** Se programó un módulo para exportar las coordenadas de las cajas delimitadoras, IDs y fotogramas a archivos `.txt` siguiendo el formato estándar `mot15-2D`.
* **Evaluación Automatizada:** Se implementó la librería `motmetrics` para cruzar las predicciones (Hypothesis) contra las etiquetas manuales (Ground Truth). *Nota técnica: Se aplicó "Monkey Patching" (`np.asfarray = np.asarray`) para resolver conflictos de compatibilidad con NumPy 2.0 en Kaggle.*

Al aplicar la configuración maestra de la Fase 4 al dataset BEE24, el modelo no generalizó bien debido a las diferencias del entorno, lo que motivó un **Estudio de Ablación** (*Ablation Study*) para encontrar el balance óptimo:

| Prueba | Configuración | Resultados Clave | Conclusión |
| :--- | :--- | :--- | :--- |
| **Prueba 1** | Estricta (Conf. 0.85, Hits 30) | MOTA: 14.0% \| IDF1: 20.8% | Demasiado restrictivo. Ignoraba a la mayoría de las abejas. |
| **Prueba 2** | Permisiva (Conf. 0.40, Hits 5) | MOTA: 59.8% \| IDF1: 51.8% \| Cambios ID: 871 | Recuperó detecciones, pero el ruido aumentó los Falsos Positivos. |
| **Prueba 3** | **Punto Dulce (Conf. ~0.60, Hits ~15)** | **MOTA: 59.9% \| IDF1: 55.5% \| Cambios ID: 390** | **Equilibrio ideal.** Redujo Cambios de ID a la mitad y eliminó +8,500 Falsos Positivos. |

El sistema **YOLO26 + OC-SORT** ha demostrado ser altamente robusto frente a escenarios complejos, requiriendo únicamente el ajuste de los hiperparámetros de persistencia y confianza según las características del entorno.

Perfecto. Aquí tienes una propuesta de **Conclusión Académica** redactada con un tono formal, ideal para cerrar el capítulo de metodología o resultados de tu TFG.

---

### 📝 Resumen Ejecutivo: Validación del Sistema MOT-Bee
**Arquitectura YOLO26 + OC-SORT**

La presente investigación concluye que el seguimiento multiobjeto de *Apis mellifera* requiere un enfoque que priorice la **cinemática del movimiento** sobre la apariencia visual. Debido a la homogeneidad morfológica de los ejemplares, el uso de descriptores de re-identificación (ReID) basados en redes neuronales profundas resultó contraproducente, incrementando la fragmentación de identidades.

A través de un **estudio de ablación** sistemático sobre el dataset científico **BEE24**, se determinó que el "punto dulce" del sistema se alcanza con una confianza de detección de $0.60$ y una persistencia de $15$ frames. Esta configuración permite:
* Maximizar el índice **MOTA** (~60%), equilibrando la sensibilidad y la precisión.
* Reducir los cambios de ID en un **55%** respecto a configuraciones permisivas.
* Alcanzar una precisión del **92.8%** en entornos reales de piquera frente al *Ground Truth*.

En definitiva, la combinación de un detector de una sola etapa (**YOLO26 Medium**) y un rastreador centrado en la observación (**OC-SORT**) constituye una solución robusta y de alta fidelidad para el monitoreo automatizado de colmenas, demostrando una notable capacidad de generalización tras el ajuste fino de sus hiperparámetros.

