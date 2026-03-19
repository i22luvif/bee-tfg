import os
import numpy as np
import pandas as pd
import argparse

# =====================================================================
# --- EL PARCHE MÁGICO (Monkey Patching) ---
# =====================================================================
# Problema: La librería científica 'motmetrics' usa una función antigua 
# (np.asfarray) que fue eliminada en las versiones modernas de Numpy (2.0+).
# Solución: En lugar de usar versiones viejas y romper otras librerías, 
# inyectamos esta función en memoria dinámicamente. Esto garantiza que el 
# evaluador funcione en cualquier ordenador moderno o nube (como Kaggle) sin fallos.
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a: np.asarray(a, dtype=float)

import motmetrics as mm

def evaluar_tracker_individual(ruta_dataset_gt, ruta_resultados_tracker):
    """
    Evalúa matemáticamente las predicciones de un tracker específico contra 
    el Ground Truth (la realidad anotada a mano por humanos).
    """
    mh = mm.metrics.create()
    acumuladores = []
    nombres_secuencias = []
    
    # Verificación de seguridad: Comprobamos que las respuestas correctas existan
    if not os.path.exists(ruta_dataset_gt):
        print(f"[!] ERROR: No existe la carpeta de Ground Truth: {ruta_dataset_gt}")
        return None

    # Listamos todas las carpetas de vídeo del dataset (SEQ_01, SEQ_02...)
    secuencias = sorted([d for d in os.listdir(ruta_dataset_gt) if os.path.isdir(os.path.join(ruta_dataset_gt, d))])
    
    for seq in secuencias:
        # Rutas a comparar: El archivo real vs el archivo generado por la IA
        ruta_gt = os.path.join(ruta_dataset_gt, seq, "gt", "gt.txt")
        ruta_hyp = os.path.join(ruta_resultados_tracker, f"{seq}.txt")
        
        if not os.path.exists(ruta_gt):
            continue
            
        # Tolerancia a fallos: Si un tracker falló en un vídeo concreto, lo saltamos
        # sin que el programa entero colapse.
        if not os.path.exists(ruta_hyp):
            print(f"[!] Faltan predicciones para la secuencia {seq} en {ruta_resultados_tracker}")
            continue
            
        # Cargamos los datos en el formato estándar MOT15/16
        gt = mm.io.loadtxt(ruta_gt, fmt='mot15-2D', min_confidence=1)
        hyp = mm.io.loadtxt(ruta_hyp, fmt='mot15-2D')
        
        # --- CÁLCULO DE INTERSECCIÓN SOBRE UNIÓN (IoU) ---
        # distth=0.5 significa que la caja de la IA tiene que solaparse al menos 
        # un 50% con la caja real para que cuente como "Acierto".
        acc = mm.utils.compare_to_groundtruth(gt, hyp, 'iou', distth=0.5)
        
        acumuladores.append(acc)
        nombres_secuencias.append(seq)
        
    if not acumuladores:
        return None
        
    # El motor computa todas las secuencias y devuelve un resumen detallado
    resumen = mh.compute_many(
        acumuladores, 
        metrics=['idf1', 'mota', 'motp', 'num_false_positives', 'num_misses', 'num_switches', 'mostly_tracked', 'mostly_lost'], 
        names=nombres_secuencias, 
        generate_overall=True # Genera una fila extra con la media ponderada de todo el dataset
    )
    return resumen

def evaluar_benchmark_completo(ruta_dataset_gt, ruta_benchmark_root):
    """
    Función de Orquestación: Entra en la carpeta de resultados, detecta todos
    los trackers que han competido, los evalúa uno a uno y consolida los
    resultados en una única tabla comparativa lista para la memoria del TFG.
    """
    print(f"[*] Iniciando evaluación de todos los trackers en: {ruta_benchmark_root}")
    
    if not os.path.exists(ruta_benchmark_root):
        print(f"[!] ERROR: No se encontró la carpeta de resultados: {ruta_benchmark_root}")
        return

    # Escaneamos qué trackers han participado (bytetrack, ocsort, strongsort...)
    trackers = sorted([d for d in os.listdir(ruta_benchmark_root) if os.path.isdir(os.path.join(ruta_benchmark_root, d))])
    
    if not trackers:
        print("[!] No hay carpetas de trackers para evaluar.")
        return

    resultados_globales = []

    # Evaluamos cada algoritmo
    for tracker_name in trackers:
        print(f"\n📊 Evaluando: {tracker_name.upper()}...")
        ruta_resultados_tracker = os.path.join(ruta_benchmark_root, tracker_name)
        
        resumen_df = evaluar_tracker_individual(ruta_dataset_gt, ruta_resultados_tracker)
        
        if resumen_df is not None:
            # --- EXTRACCIÓN DE DATOS CLAVE ---
            # No queremos los datos de cada vídeo individual, solo queremos la nota 
            # media global ('OVERALL') para poder comparar los algoritmos entre sí.
            overall_metrics = resumen_df.loc['OVERALL'].copy()
            overall_metrics['Tracker'] = tracker_name.upper()
            resultados_globales.append(overall_metrics)
        else:
            print(f"  [!] No se pudieron calcular métricas para {tracker_name}")

    if not resultados_globales:
        print("No se generaron resultados finales.")
        return

    # Juntamos todos los resultados en una única tabla de Pandas
    tabla_final = pd.DataFrame(resultados_globales)
    
    # Reordenamos estéticamente para que el nombre del 'Tracker' sea la primera columna
    columnas = ['Tracker'] + [c for c in tabla_final.columns if c != 'Tracker']
    tabla_final = tabla_final[columnas]
    
    # Renombramos las variables técnicas a la nomenclatura estándar académica 
    # para que luzca profesional en el documento del TFG.
    tabla_final.rename(columns={
        'idf1': 'IDF1', 'mota': 'MOTA', 'motp': 'MOTP',
        'num_false_positives': 'FP', 
        'num_misses': 'FN',           # Falsos Negativos (Abejas perdidas)
        'num_switches': 'IDsw',       # Cambios de identidad (Errores del tracker)
        'mostly_tracked': 'MT',       # Trayectorias completadas con éxito
        'mostly_lost': 'ML'           # Trayectorias perdidas muy rápido
    }, inplace=True)

    print("\n" + "="*85)
    print("🏆 RESULTADOS FINALES DEL BENCHMARK (Métricas MOT)")
    print("="*85)
    
    # Formateamos los decimales a porcentajes legibles (ej: 0.85 -> 85.0%)
    formatos = {'IDF1': '{:.1%}', 'MOTA': '{:.1%}', 'MOTP': '{:.3f}'}
    print(tabla_final.to_string(index=False, formatters=formatos))
    print("="*85)
    
    # --- EXPORTACIÓN A CSV ---
    # Guarda la tabla final automáticamente para evitar errores de transcripción manual.
    # Puedes abrir este archivo con Excel y copiar/pegar la tabla en Word/LaTeX.
    ruta_csv = os.path.join(ruta_benchmark_root, "resultados_finales_tfg.csv")
    tabla_final.to_csv(ruta_csv, index=False)
    print(f"\n✅ Tabla guardada en Excel/CSV en: {ruta_csv}")
    print("   (Lista para copiar y pegar en la memoria del TFG)")

# Punto de entrada desde terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluador Masivo de Trackers (MOT)")
    
    # Ruta al Ground Truth (La respuesta correcta)
    parser.add_argument("--gt", type=str, default="datasets/raw/BEE24/test", 
                        help="Ruta al Ground Truth del dataset BEE24")
    
    # Ruta raíz donde están almacenadas todas las carpetas generadas por los trackers
    parser.add_argument("--benchmark_dir", type=str, default="runs/benchmark_results", 
                        help="Carpeta que contiene las subcarpetas de cada tracker")
    
    args = parser.parse_args()
    evaluar_benchmark_completo(args.gt, args.benchmark_dir)