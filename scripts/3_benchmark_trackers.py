import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import create_tracker

def realizar_tracking(frame: np.ndarray, model: YOLO, tracker) -> list:
    """ 
    Orquesta la detección espacial y la asociación temporal fotograma a fotograma.
    Realiza la inferencia con YOLO y transfiere los tensores al algoritmo de seguimiento.
    """
    # 1. Inferencia Espacial: Umbrales empíricos estrictos para minimizar Falsos Positivos
    res = model(frame, verbose=False, conf=0.85, iou=0.5)
    
    if len(res[0].boxes) > 0:
        # [ESTRATEGIA TÉCNICA 1]: Formateo de Tensores para BoxMOT
        # BoxMOT requiere estrictamente un array numpy (N, 6) apilado horizontalmente:
        # [x_min, y_min, x_max, y_max, conf, class_id]
        detecciones = np.hstack((
            res[0].boxes.xyxy.cpu().numpy(),                 
            res[0].boxes.conf.cpu().numpy().reshape(-1, 1),  
            res[0].boxes.cls.cpu().numpy().reshape(-1, 1)    
        ))
        
        # [ESTRATEGIA TÉCNICA 2]: Escudo de Tolerancia a Fallos (Kalman Filter)
        # Algoritmos complejos (ej. DeepOCSORT) pueden sufrir colapsos matemáticos (LinAlgError)
        # si la matriz de covarianza de una detección resulta no definida positiva.
        try:
            return tracker.update(detecciones, frame)
        except Exception:
            # Si el tracker falla en este frame puntual, devolvemos lista vacía para no detener la ejecución
            return []
    
    return []

def run_benchmark(model_path: str, input_path: str, output_root: str) -> None:
    """
    Motor de Inferencia Multi-Algoritmo (Stream-Agnostic).
    Evalúa secuencialmente modelos de seguimiento del Estado del Arte sobre un dataset.
    """
    # Selección de algoritmos SOTA (State of the Art) a evaluar
    trackers_to_test = ['bytetrack', 'botsort', 'ocsort', 'deepocsort', 'strongsort']
    
    print(f"[*] Instanciando detector base YOLO: {model_path}")
    model = YOLO(model_path)
    
    elementos = sorted([d for d in os.listdir(input_path)])
    if not elementos:
        print(f"[!] Directorio de entrada vacío o inexistente: {input_path}")
        return

    # [ESTRATEGIA TÉCNICA 3]: Optimización de Hardware (Half-Precision)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    is_half = True if device == 'cuda:0' else False

    for tracker_type in trackers_to_test:
        print(f"\n🚀 EJECUTANDO PIPELINE DE SEGUIMIENTO: {tracker_type.upper()}")
        
        tracker_out_dir = os.path.join(output_root, tracker_type)
        os.makedirs(tracker_out_dir, exist_ok=True)

        for elem in elementos:
            ruta_full = os.path.join(input_path, elem)
            print(f"  🎬 Procesando secuencia: {elem}")
            
            # Inicialización de Tracker + Extractor de Características (ReID)
            # Se inyecta OSNet (osnet_x0_25_msmt17.pt) para extraer firmas visuales
            tracker = create_tracker(tracker_type, None, "osnet_x0_25_msmt17.pt", device, half=is_half)
            results_mot = []

            # ==========================================================
            # [ESTRATEGIA TÉCNICA 4]: Ingesta de Datos Agnóstica (Folder vs Video)
            # ==========================================================
            if os.path.isdir(ruta_full):
                # Flujo A: Formato Académico/Científico (Dataset secuencial tipo MOT16)
                ruta_img = os.path.join(ruta_full, "img1") if os.path.exists(os.path.join(ruta_full, "img1")) else ruta_full
                fotos = sorted([f for f in os.listdir(ruta_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                for i, nombre_foto in enumerate(fotos):
                    frame = cv2.imread(os.path.join(ruta_img, nombre_foto))
                    if frame is None: continue
                    
                    tracks = realizar_tracking(frame, model, tracker)
                    
                    # [ESTRATEGIA TÉCNICA 5]: Estandarización de Salida (Conversión a MOT16)
                    for t in tracks:
                        x1, y1, x2, y2, track_id = t[0], t[1], t[2], t[3], int(t[4])
                        # Transformación de coordenadas: de (x1, y1, x2, y2) a (x_min, y_min, width, height)
                        results_mot.append(f"{i+1},{track_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1")
            
            else:
                # Flujo B: Formato de Producción/Campo (Archivos multimedia .mp4, .avi)
                cap = cv2.VideoCapture(ruta_full)
                f_idx = 1
                while True:
                    ret, frame = cap.read()
                    if not ret: break 
                    
                    tracks = realizar_tracking(frame, model, tracker)
                    
                    for t in tracks:
                        x1, y1, x2, y2, track_id = t[0], t[1], t[2], t[3], int(t[4])
                        results_mot.append(f"{f_idx},{track_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1")
                    f_idx += 1
                cap.release()

            # Persistencia de resultados en formato estandarizado
            nombre_txt = os.path.splitext(elem)[0]
            txt_path = os.path.join(tracker_out_dir, f"{nombre_txt}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(results_mot))
            print(f"    ✅ Trazas exportadas: {txt_path}")

    print(f"\n✅ BENCHMARK FINALIZADO. Resultados almacenados en: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecución de Benchmark para algoritmos SOTA de Multiple Object Tracking")
    
    parser.add_argument("--model", type=str, default="model/best_bee_medium.pt", help="Ruta al modelo YOLO detector")
    parser.add_argument("--input", type=str, default="datasets/raw/BEE24/test", help="Directorio con secuencias/vídeos a evaluar")
    parser.add_argument("--output", type=str, default="runs/benchmark_results", help="Directorio de volcado de resultados MOT")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.input, args.output)