
import streamlit as st
import cv2
import os
from ultralytics import YOLO

# Configuramos la página web para que ocupe todo el ancho de la pantalla.
# Por defecto, Streamlit centra todo en una columna estrecha. Al procesar vídeo, necesitamos espacio visual.
st.set_page_config(page_title="Detector de Abejas", layout="wide") 

# Ponemos el título principal en la parte superior de la web.
st.title("TFG - Panel Avanzado de Detección 🐝")

# ==========================================
# ZONA 1: LA BARRA LATERAL (FRONTEND)
# ==========================================

# Creamos un título específico para el menú de la izquierda (sidebar).
st.sidebar.header("⚙️ Hiperparámetros de YOLO")

# Creamos un desplegable para elegir el archivo del modelo entrenado.
modelo_elegido = st.sidebar.selectbox("Modelo de IA:", ("model/best_bee.pt",))

# Creamos cajas numéricas para que el usuario introduzca la Confianza y el IoU.
confianza = st.sidebar.number_input("Confianza mínima (conf):", min_value=0.01, max_value=1.0, value=0.40, placeholder="Type a number...")
iou_valor = st.sidebar.number_input("Solapamiento máximo (IoU):", min_value=0.01, max_value=1.0, value=0.50, placeholder="Type a number...")

# Desplegable para el tamaño al que YOLO redimensionará la imagen antes de pensar.
tamano_img = st.sidebar.selectbox("Resolución de análisis (imgsz):", (320, 640, 1024), index=1)

# Dibuja una línea horizontal gris. 
st.sidebar.divider() 

# Menú para elegir el algoritmo de rastreo (Tracker).
st.sidebar.header("🎯 Configuración del Tracker")
tracker_elegido = st.sidebar.selectbox("Algoritmo de Rastreo:", ("botsort.yaml", "bytetrack.yaml"))
st.sidebar.info("💡 ByteTrack es más rápido, pero BoT-SORT es mejor recuperando abejas que se ocultan detrás de otras.")


# ==========================================
# ZONA 2: PREPARACIÓN DE VÍDEOS (BACKEND + FRONTEND)
# ==========================================

# Guardamos la ruta de la carpeta donde están nuestros vídeos brutos.
carpeta_videos = "data/raw"

# Leemos la carpeta de Windows y sacamos una lista con los nombres de los vídeos que hay dentro.
# Si la carpeta existe, saca la lista (`os.listdir`). Si no existe (para evitar errores), devuelve una lista vacía `[]`.
videos_disponibles = os.listdir(carpeta_videos) if os.path.exists(carpeta_videos) else []

# Desplegable en la pantalla principal que muestra los vídeos encontrados.
video_elegido = st.selectbox("🎥 Selecciona el video a analizar:", videos_disponibles)

# Creamos el botón gigante. Todo lo que esté indentado debajo de este 'if' SOLO ocurrirá al hacer clic.
if st.button("¡Iniciar Análisis Avanzado!"):
    
    # Comprobación de seguridad.
    # Si el usuario pulsa el botón pero la carpeta de vídeos estaba vacía, evitamos que el programa explote.
    if not video_elegido:
        st.error("⚠️ No se ha encontrado ningún video. Pon uno en la carpeta data/raw.")
    else:
        
        # ==========================================
        # ZONA 3: EL MOTOR DEL PROGRAMA (BACKEND)
        # ==========================================

        # Unimos el nombre de la carpeta y el del vídeo (ej: "data/raw" + "video.mp4" = "data/raw/video.mp4")
        ruta_video = os.path.join(carpeta_videos, video_elegido)
        
        # Cargamos la red neuronal en la memoria del ordenador.
        modelo = YOLO(modelo_elegido)
        
        # Le decimos a OpenCV que abra el archivo de vídeo para empezar a extraer fotos.
        video = cv2.VideoCapture(ruta_video)
        
        # Definimos dónde y cómo se va a guardar el vídeo con los recuadros dibujados.
        ruta_salida = "runs/detect/video_procesado.mp4"
        
        # Extraemos las medidas exactas (ancho, alto y fotogramas por segundo) del vídeo original.
        # La grabadora necesita saber de qué tamaño debe crear el vídeo nuevo para que no se deforme.
        ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) #Casteamos a int porque get devuelve un floa
        alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        # Configuramos el codificador de vídeo (el formato) y encendemos la "grabadora" (VideoWriter).
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        salida = cv2.VideoWriter(ruta_salida, codec, fps, (ancho, alto))
        
        # Preparamos un espacio en blanco en la web donde luego meteremos el vídeo.
        marco_video = st.empty()
        st.info("Procesando video con ajustes avanzados...")

        
        # Creamos un "conjunto" matemático vacío para llevar la cuenta.
        # Un 'set' ignora automáticamente los números repetidos. Perfecto para contar DNI únicos de abejas.
        abejas_unicas = set()
        
        # Bucle infinito. Se repetirá una vez por cada fotograma del vídeo.
        while True:
            # Leemos un fotograma. 'exito' nos dirá True si hay foto, o False si el vídeo se acabó.
            exito, fotograma = video.read()
            
            # Si el vídeo se ha terminado...
            if not exito:
                # Mostramos el mensaje final de éxito en la web y rompemos (break) el bucle infinito.
                st.success(f"¡Análisis completado! Se han detectado {len(abejas_unicas)} abejas únicas en total.")
                break
            
            # Pasamos la foto y los parámetros que el usuario eligió en la web a YOLO.
            # persist=True es vital para que el Tracker no olvide a las abejas del fotograma anterior.
            resultados = modelo.track(
                fotograma, 
                conf=confianza, 
                iou=iou_valor,       
                imgsz=tamano_img,    
                tracker=tracker_elegido, 
                persist=True
            )
            
            # Le pedimos a YOLO que nos devuelva la foto, pero ya con los recuadros y números pintados encima.
            fotograma_dibujado = resultados[0].plot()
            
            # Comprobamos si la IA ha detectado al menos una abeja y le ha puesto DNI (id).
            # Si no hay abejas en este fotograma y no hiciéramos esta comprobación, el código daría error.
            if resultados[0].boxes.id is not None:
                # Extraemos los IDs de la IA, los convertimos a números normales (int) de Python.
                ids_detectados = resultados[0].boxes.id.cpu().numpy().astype(int)
                # Metemos esos números en nuestro conjunto matemático (él solo descartará los repetidos).
                abejas_unicas.update(ids_detectados)
            
            # Preparamos el texto del contador (ej: "Abejas Totales: 15").
            texto_contador = f"Abejas Totales: {len(abejas_unicas)}"
            # Usamos OpenCV para escribir ese texto en verde neón en la esquina de la foto.
            cv2.putText(fotograma_dibujado, texto_contador, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Guardamos esta foto pintada dentro del archivo MP4 final en el disco duro.
            salida.write(fotograma_dibujado)
            
            # Convertimos los colores de la foto de formato BGR (que usa OpenCV) a RGB (que usa internet).
            fotograma_rgb = cv2.cvtColor(fotograma_dibujado, cv2.COLOR_BGR2RGB)
            # Mostramos la foto a todo color dentro del marco en blanco que creamos en nuestra web.
            marco_video.image(fotograma_rgb, channels="RGB")
            
        # ==========================================
        # ZONA 4: LIMPIEZA Y DESCARGA (CIERRE)
        # ==========================================

        # Apagamos el lector de vídeo original y cerramos la "grabadora".
        # Si no lo cerramos, el archivo MP4 se quedará corrupto y no se podrá reproducir.
        video.release()
        salida.release()
        
        # Abrimos el vídeo procesado que acabamos de guardar en modo "lectura binaria" ('rb').
        with open(ruta_salida, "rb") as archivo_video:
            # Invocamos el botón mágico de Streamlit que permite al usuario bajarse ese archivo.
            st.download_button(
                label="⬇️ Descargar Video Procesado",
                data=archivo_video,
                file_name=f"procesado_{video_elegido}",
                mime="video/mp4"
            )
            
            

"""DUDAS RESUELTAS"""

"""
* ids_detectados = resultados[0].boxes.id.cpu().numpy().astype(int)
Utilizo esta cadena de métodos para extraer los tensores matemáticos de la GPU, convertirlos a matrices estándar de Python y asegurar que los identificadores sean números enteros y no decimales".



* fotograma_rgb = cv2.cvtColor(fotograma_dibujado, cv2.COLOR_BGR2RGB)
CV2 trabaja con GBR y las paginas web de hoy en dia con RGB


* with open(ruta_salida, "rb") as archivo_video:
Al poner "rb" (Read Binary), le estamos diciendo a Python: "Oye, no intentes entender ni leer palabras en este archivo. Simplemente coge el bloque de unos y ceros tal y como está, y entrégaselo a Streamlit para que el usuario pueda descargarlo en su ordenador". Si usaras solo "r", Python daría un error fatal al intentar decodificar el vídeo como si fuera un libro
"""