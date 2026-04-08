import streamlit as st
import subprocess
import os
import sys

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Bee-TFG Pro", page_icon="🐝", layout="wide")

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
entorno = os.environ.copy()
entorno["PYTHONIOENCODING"] = "utf-8"
entorno["PYTHONUTF8"] = "1"

def ejecutar_script_en_vivo(comando, descripcion):
    """Ejecuta un comando rápido y muestra la terminal en vivo."""
    st.markdown(f"### ⚙️ Ejecutando: {descripcion}")
    st.caption("💡 *Nota: Para detener este proceso en vivo, pulsa el botón 'Stop' arriba a la derecha en Streamlit.*")
    caja_terminal = st.empty()
    log_texto = ""
    
    proceso = subprocess.Popen(
        comando, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8", 
        errors="replace", 
        env=entorno
    )
    
    for linea in proceso.stdout:
        log_texto += linea
        lineas_visibles = "\n".join(log_texto.splitlines()[-25:])
        caja_terminal.code(lineas_visibles, language="bash")
        
    proceso.wait() 
    if proceso.returncode == 0:
        st.success(f"✅ {descripcion} completado con éxito.")
    else:
        # Si el usuario lo para a la fuerza o falla
        st.error(f"❌ Proceso interrumpido o fallido: {descripcion}.")

def lanzar_en_segundo_plano(comando, ruta_log):
    """Inicia un script pesado en segundo plano."""
    os.makedirs(os.path.dirname(ruta_log), exist_ok=True)
    archivo = open(ruta_log, "w", encoding="utf-8", errors="replace")
    subprocess.Popen(
        comando, 
        stdout=archivo, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8", 
        errors="replace", 
        env=entorno
    )

def leer_log_en_vivo(ruta_log):
    """Lee las últimas líneas del archivo log."""
    if os.path.exists(ruta_log):
        with open(ruta_log, "r", encoding="utf-8", errors="replace") as f:
            lineas = f.readlines()
            return "".join(lineas[-40:])
    return "El archivo log aún no se ha creado o está vacío."

def detener_proceso_windows(nombre_script):
    """Busca un script de Python ejecutándose en Windows y lo mata a la fuerza."""
    # Comando de Windows (WMIC) para matar procesos por el nombre de su argumento
    comando_kill = f'wmic process where "commandline like \'%{nombre_script}%\' and name=\'python.exe\'" delete >nul 2>&1'
    os.system(comando_kill)

# ==========================================
# 3. INTERFAZ WEB (MENÚ LATERAL)
# ==========================================

st.sidebar.title("🐝 Menú de Navegación")
st.sidebar.write("Selecciona el módulo:")

opcion = st.sidebar.radio(
    "Pipeline TFG:",
    [
        "📁 1. Preparar Dataset",
        "🚀 2. Entrenar Modelo (YOLO)",
        "🏃‍♂️ 3. Benchmark Trackers",
        "📊 4. Evaluar Trackers (MOT)",
        "✨ 5. Pre-Anotación Mágica"
    ]
)

st.sidebar.divider()

# BOTÓN GLOBAL DE EMERGENCIA EN LA BARRA LATERAL
st.sidebar.subheader("🚨 Control de Emergencia")
if st.sidebar.button("🛑 Detener TODOS los procesos", type="primary"):
    scripts_a_matar = ["1_prepare_data.py", "2_train.py", "3_benchmark_trackers.py", "4_evaluate_tracking.py", "0_pre_annotate.py"]
    for script in scripts_a_matar:
        detener_proceso_windows(script)
    st.sidebar.success("Procesos detenidos a la fuerza.")


# --- PANTALLA PRINCIPAL ---
st.title("🎛️ Panel de Control del Sistema")
st.divider()

if opcion == "📁 1. Preparar Dataset":
    st.header("1. Preparación de Datos")
    st.write("Unifica tus fotos, previene la fuga de datos y formatea todo al estándar YOLOv8.")
    st.write("<br>", unsafe_allow_html=True)
    
    if st.button("▶️ Ejecutar Formateo de Dataset", use_container_width=True, type="primary"):
        comando = [sys.executable, "scripts/1_prepare_data.py", "--input", "datasets/raw/mendeley_dataset/detection", "--output", "datasets/ready_for_yolo/mendeley_yolo"]
        ejecutar_script_en_vivo(comando, "Preparación de Datos")

elif opcion == "🚀 2. Entrenar Modelo (YOLO)":
    st.header("2. Entrenamiento en Segundo Plano")
    st.write("Proceso de alto coste computacional. Se lanzará de forma asíncrona.")
    ruta_log_entrenamiento = "runs/entrenamiento_actual.log"
    st.write("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔥 Iniciar Entrenamiento", type="primary", use_container_width=True):
            comando_train = [sys.executable, "scripts/2_train.py", "--data", "datasets/ready_for_yolo/mendeley_yolo/data.yaml", "--epochs", "150"]
            lanzar_en_segundo_plano(comando_train, ruta_log_entrenamiento)
            st.success("¡Entrenamiento iniciado en el servidor!")
            
    with col2:
        if st.button("🔄 Actualizar", use_container_width=True):
            pass 
            
    with col3:
        # BOTÓN ESPECÍFICO PARA MATAR EL ENTRENAMIENTO
        if st.button("🛑 Detener", use_container_width=True):
            detener_proceso_windows("2_train.py")
            with open(ruta_log_entrenamiento, "a", encoding="utf-8") as f:
                f.write("\n\n[!] ENTRENAMIENTO ABORTADO POR EL USUARIO.\n")
            st.warning("Entrenamiento cancelado.")
            
    st.subheader("Terminal en vivo:")
    progreso = leer_log_en_vivo(ruta_log_entrenamiento)
    st.code(progreso, language="bash")

elif opcion == "🏃‍♂️ 3. Benchmark Trackers":
    st.header("3. Inferencia y Tracking (Benchmark)")
    st.write("Procesa los vídeos de prueba utilizando tu modelo YOLO y algoritmos del estado del arte.")
    st.write("<br>", unsafe_allow_html=True)
    
    if st.button("▶️ Ejecutar Motores de Seguimiento", use_container_width=True, type="primary"):
        comando = [sys.executable, "scripts/3_benchmark_trackers.py", "--model", "model/best_bee_medium.pt", "--input", "datasets/raw/BEE24/test", "--output", "runs/benchmark_results"]
        ejecutar_script_en_vivo(comando, "Benchmark de Trackers MOT")

elif opcion == "📊 4. Evaluar Trackers (MOT)":
    st.header("4. Evaluación Matemática")
    st.write("Cruza las cajas generadas con el Ground Truth para obtener métricas científicas.")
    st.write("<br>", unsafe_allow_html=True)
    
    if st.button("▶️ Ejecutar Evaluación Matemática", use_container_width=True, type="primary"):
        comando = [sys.executable, "scripts/4_evaluate_tracking.py", "--gt", "datasets/raw/BEE24/test", "--benchmark_dir", "runs/benchmark_results"]
        ejecutar_script_en_vivo(comando, "Evaluador MOT")

elif opcion == "✨ 5. Pre-Anotación Mágica":
    st.header("🪄 Herramienta de Pre-Anotación")
    st.write("Automatiza el etiquetado de vídeos crudos (Frame Skipping y pseudo-etiquetado).")
    st.write("<br>", unsafe_allow_html=True)
    
    video_subido = st.text_input("Ruta del vídeo a pre-anotar:", value="data/raw/mi_video.mp4")

    if st.button("✨ Iniciar Auto-Etiquetado", type="primary", use_container_width=True):
        comando = [sys.executable, "scripts/0_pre_annotate.py", "--video", video_subido]
        ejecutar_script_en_vivo(comando, f"Procesando {video_subido}...")