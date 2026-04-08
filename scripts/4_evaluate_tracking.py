import os
import numpy as np
import pandas as pd
import argparse

# =====================================================================
# [ESTRATEGIA TÉCNICA 1]: Resolución de Dependencias Dinámica (Monkey Patching)
# =====================================================================
# La librería estándar 'motmetrics' depende de una función (np.asfarray) 
# deprecada en Numpy 2.0+. Para garantizar la reproducibilidad del TFG en 
# infraestructuras modernas (nube, contenedores) sin forzar un 'downgrade' 
# de librerías core, inyectamos el método faltante directamente en el espacio 
# de nombres de Numpy en tiempo de ejecución.
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a: np.asarray(a, dtype=float)

import motmetrics as mm

def evaluar_tracker_individual(ruta_dataset_gt: str, ruta_resultados_tracker: str) -> pd.DataFrame:
    """
    Evalúa cuantitativamente las predicciones de trayectorias de un algoritmo 
    específico contra las anotaciones manuales (Ground Truth).
    """
    mh = mm.metrics.create()
    acumuladores = []
    nombres_secuencias = []
    
    if not os.path.exists(ruta_dataset_gt):
        print(f"[!] ERROR CRÍTICO: Directorio Ground Truth inaccesible: {ruta_dataset_gt}")
        return None

    # Extracción de secuencias válidas del dataset de validación/test
    secuencias = sorted([d for d in os.listdir(ruta_dataset_gt) if os.path.isdir(os.path.join(ruta_dataset_gt, d))])
    
    for seq in secuencias:
        ruta_gt = os.path.join(ruta_dataset_gt, seq, "gt", "gt.txt")
        ruta_hyp = os.path.join(ruta_resultados_tracker, f"{seq}.txt")
        
        if not os.path.exists(ruta_gt):
            continue
            
        # Tolerancia a Fallos: Omite secuencias donde el inferenciador colapsó, 
        # permitiendo evaluar el rendimiento sobre el resto del dataset.
        if not os.path.exists(ruta_hyp):
            print(f"[!] Advertencia: Ausencia de hipótesis predictivas para {seq} en {ruta_resultados_tracker}")
            continue
            
        # Parseo de tensores bajo el estándar MOT15/16
        gt = mm.io.loadtxt(ruta_gt, fmt='mot15-2D', min_confidence=1)
        hyp = mm.io.loadtxt(ruta_hyp, fmt='mot15-2D')
        
        # [ESTRATEGIA TÉCNICA 2]: Métrica de Asignación Espacial (IoU Thresholding)
        # Se define un umbral de solapamiento del 50% (distth=0.5). La matriz de 
        # costes de asociación penalizará cualquier predicción que no comparta 
        # al menos la mitad de su área con la ubicación real de la abeja.
        acc = mm.utils.compare_to_groundtruth(gt, hyp, 'iou', distth=0.5)
        
        acumuladores.append(acc)
        nombres_secuencias.append(seq)
        
    if not acumuladores:
        return None
        
    # [ESTRATEGIA TÉCNICA 3]: Agregación Global de Métricas
    # Calcula el rendimiento global ponderado ('OVERALL') de todo el dataset,
    # vital para evitar sesgos en vídeos particularmente fáciles o difíciles.
    resumen = mh.compute_many(
        acumuladores, 
        metrics=['idf1', 'mota', 'motp', 'num_false_positives', 'num_misses', 'num_switches', 'mostly_tracked', 'mostly_lost'], 
        names=nombres_secuencias, 
        generate_overall=True 
    )
    return resumen

def evaluar_benchmark_completo(ruta_dataset_gt: str, ruta_benchmark_root: str) -> None:
    """
    Función Orquestadora: Analiza el directorio de inferencias, extrae las métricas
    individuales de cada algoritmo participante y consolida los resultados en una
    matriz de rendimiento comparativa.
    """
    print(f"[*] Inicializando auditoría de métricas MOT en: {ruta_benchmark_root}")
    
    if not os.path.exists(ruta_benchmark_root):
        print(f"[!] ERROR: Directorio de inferencias no localizado: {ruta_benchmark_root}")
        return

    trackers = sorted([d for d in os.listdir(ruta_benchmark_root) if os.path.isdir(os.path.join(ruta_benchmark_root, d))])
    
    if not trackers:
        print("[!] No se detectaron algoritmos a evaluar.")
        return

    resultados_globales = []

    for tracker_name in trackers:
        print(f"\n📊 Procesando métricas para: {tracker_name.upper()}...")
        ruta_resultados_tracker = os.path.join(ruta_benchmark_root, tracker_name)
        
        resumen_df = evaluar_tracker_individual(ruta_dataset_gt, ruta_resultados_tracker)
        
        if resumen_df is not None:
            # Aislamiento de la fila agregada (OVERALL) para la comparativa macro
            overall_metrics = resumen_df.loc['OVERALL'].copy()
            overall_metrics['Tracker'] = tracker_name.upper()
            resultados_globales.append(overall_metrics)
        else:
            print(f"  [!] Fallo en el cómputo de métricas para {tracker_name}")

    if not resultados_globales:
        print("Auditoría abortada: Ausencia de datos válidos.")
        return

    # [ESTRATEGIA TÉCNICA 4]: Construcción del DataFrame Final y Estandarización
    tabla_final = pd.DataFrame(resultados_globales)
    
    columnas = ['Tracker'] + [c for c in tabla_final.columns if c != 'Tracker']
    tabla_final = tabla_final[columnas]
    
    tabla_final.rename(columns={
        'idf1': 'IDF1', 'mota': 'MOTA', 'motp': 'MOTP',
        'num_false_positives': 'FP', 
        'num_misses': 'FN',           
        'num_switches': 'IDsw',       
        'mostly_tracked': 'MT',       
        'mostly_lost': 'ML'           
    }, inplace=True)

    print("\n" + "="*85)
    print("🏆 RESULTADOS FINALES DEL BENCHMARK (Métricas MOT)")
    print("="*85)
    
    formatos = {'IDF1': '{:.1%}'.format, 'MOTA': '{:.1%}'.format, 'MOTP': '{:.3f}'.format}
    print(tabla_final.to_string(index=False, formatters=formatos))
    print("="*85)
    
    # Persistencia de la matriz en disco duro para análisis post-hoc
    ruta_csv = os.path.join(ruta_benchmark_root, "resultados_finales_tfg.csv")
    tabla_final.to_csv(ruta_csv, index=False)
    print(f"\n✅ Matriz de resultados exportada con éxito a: {ruta_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditor Masivo de Algoritmos MOT (Multiple Object Tracking)")
    
    parser.add_argument("--gt", type=str, default="datasets/raw/BEE24/test", 
                        help="Directorio origen del Ground Truth (Anotaciones expertas)")
    parser.add_argument("--benchmark_dir", type=str, default="runs/benchmark_results", 
                        help="Directorio con las inferencias de los modelos a evaluar")
    
    args = parser.parse_args()
    evaluar_benchmark_completo(args.gt, args.benchmark_dir)