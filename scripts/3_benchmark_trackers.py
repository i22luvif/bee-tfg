import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import create_tracker

def realizar_tracking(frame, model, tracker):
    """ 
    Función auxiliar que centraliza la detección y el seguimiento.
    Recibe un fotograma, usa YOLO para ver dónde están las abejas, 
    y le pasa esa información al Tracker para que les asigne o mantenga un ID.
    """
    # 1. Inferencia: Buscamos abejas. 
    # Usamos conf=0.85 y iou=0.5 (los hiperparámetros óptimos encontrados empíricamente)
    res = model(frame, verbose=False, conf=0.85, iou=0.5)
    
    # Si YOLO ha encontrado al menos una abeja en este fotograma...
    if len(res[0].boxes) > 0:
        # 2. Formateo de datos: La librería BoxMOT exige un formato muy específico.
        # Necesita un array de numpy apilado horizontalmente (hstack) con esta estructura:
        # [Coordenada_X1, Coordenada_Y1, Coordenada_X2, Coordenada_Y2, Confianza, Clase]
        detecciones = np.hstack((
            res[0].boxes.xyxy.cpu().numpy(),                 # Coordenadas
            res[0].boxes.conf.cpu().numpy().reshape(-1, 1),  # % de Confianza
            res[0].boxes.cls.cpu().numpy().reshape(-1, 1)    # Clase (0 para 'bee')
        ))
        
        # 3. El Tracker analiza el movimiento y devuelve las cajas con su ID (matrícula) asignado
        return tracker.update(detecciones, frame)
    
    # Si no hay abejas en este frame, devolvemos una lista vacía
    return []

def run_benchmark(model_path, input_path, output_root):
    """
    Motor de Inferencia Multi-Algoritmo.
    Ejecuta una carrera comparativa entre 5 trackers usando el mismo modelo YOLO
    como detector. Es agnóstico a la fuente: procesa tanto carpetas de fotos como vídeos.
    """
    # 1. Los 5 algoritmos del estado del arte que vamos a evaluar en el TFG
    # (Actualizado a 'strongsort' por su uso nativo de Re-Identificación)
    trackers_to_test = ['bytetrack', 'botsort', 'ocsort', 'deepocsort', 'strongsort']
    
    print(f"[*] Cargando modelo entrenado YOLO: {model_path}")
    model = YOLO(model_path)
    
    # Identificamos el contenido de la carpeta de entrada (ej: SEQ_01, SEQ_02, video.mp4...)
    elementos = sorted([d for d in os.listdir(input_path)])
    if not elementos:
        print(f"[!] No se encontró nada en {input_path}")
        return

    # --- CONFIGURACIÓN DE HARDWARE PARA PYTORCH/BOXMOT ---
    # A diferencia de YOLO que acepta un '0', BoxMOT exige la nomenclatura exacta de PyTorch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Si tenemos GPU, activamos la precisión media (Float16) para que los cálculos matemáticos
    # del tracker vayan el doble de rápido sin perder precisión. La CPU no soporta esto bien.
    is_half = True if device == 'cuda:0' else False

    # Bucle Principal: Evaluamos un tracker tras otro
    for tracker_type in trackers_to_test:
        print(f"\n🚀 PROBANDO TRACKER: {tracker_type.upper()}")
        
        # Creamos una carpeta para los resultados de este tracker concreto
        tracker_out_dir = os.path.join(output_root, tracker_type)
        os.makedirs(tracker_out_dir, exist_ok=True)

        # Bucle Secundario: Procesamos todos los vídeos/secuencias con el tracker actual
        for elem in elementos:
            ruta_full = os.path.join(input_path, elem)
            print(f"  🎬 Procesando: {elem}")
            
            # Inicializamos el tracker en memoria.
            # Aquí le inyectamos "osnet_x0_25_msmt17.pt", el modelo ultraligero de Re-Identificación
            # que extrae firmas visuales de las abejas para que StrongSORT/DeepOCSORT no pierdan el rastro.
            tracker = create_tracker(tracker_type, None, "osnet_x0_25_msmt17.pt", device, half=is_half)
            results_mot = []

            # ==========================================================
            # --- LÓGICA HÍBRIDA (STREAM-AGNOSTIC) ---
            # ==========================================================
            
            if os.path.isdir(ruta_full):
                # CASO A: ESCENARIO CIENTÍFICO (BEE24) - Es una carpeta con fotos
                # Busca la subcarpeta "img1" (estándar MOT) o lee directamente de la carpeta
                ruta_img = os.path.join(ruta_full, "img1") if os.path.exists(os.path.join(ruta_full, "img1")) else ruta_full
                fotos = sorted([f for f in os.listdir(ruta_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                for i, nombre_foto in enumerate(fotos):
                    frame = cv2.imread(os.path.join(ruta_img, nombre_foto))
                    if frame is None: continue
                    
                    tracks = realizar_tracking(frame, model, tracker)
                    
                    # Formateo al estándar oficial MOT16
                    for t in tracks:
                        x1, y1, x2, y2, track_id = t[0], t[1], t[2], t[3], int(t[4])
                        # El estándar MOT no usa [x2, y2], sino [ancho, alto]. Hacemos la resta matemática:
                        results_mot.append(f"{i+1},{track_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1")
            
            else:
                # CASO B: ESCENARIO DE CAMPO - Es un archivo de vídeo comprimido (.mp4)
                cap = cv2.VideoCapture(ruta_full)
                f_idx = 1
                while True:
                    ret, frame = cap.read()
                    if not ret: break # Si el vídeo termina, salimos del bucle
                    
                    tracks = realizar_tracking(frame, model, tracker)
                    
                    # Formateo al estándar oficial MOT16
                    for t in tracks:
                        x1, y1, x2, y2, track_id = t[0], t[1], t[2], t[3], int(t[4])
                        results_mot.append(f"{f_idx},{track_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1")
                    f_idx += 1
                cap.release()
            # ==========================================================

            # Guardamos la lista de trayectorias en un archivo .txt con el mismo nombre que la secuencia
            nombre_txt = os.path.splitext(elem)[0]
            txt_path = os.path.join(tracker_out_dir, f"{nombre_txt}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(results_mot))
            print(f"    ✅ TXT generado: {txt_path}")

    print(f"\n✅ BENCHMARK FINALIZADO. Resultados en: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta comparativa de trackers de estado del arte")
    
    # Parámetros genéricos para facilitar su ejecución desde consola
    parser.add_argument("--model", type=str, default="model/best_bee_medium.pt", help="Ruta al modelo YOLO entrenado")
    parser.add_argument("--input", type=str, default="datasets/raw/BEE24/test", help="Carpeta de vídeos o secuencias MOT")
    parser.add_argument("--output", type=str, default="runs/benchmark_results", help="Directorio para guardar los txt")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.input, args.output)