import argparse
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import create_tracker

def realizar_tracking(frame: np.ndarray, model: YOLO, tracker) -> list:
    """
    Ejecuta detección con YOLO y actualiza el tracker sobre un único fotograma.
    """
    # Detección pura con YOLO
    inference_results = model(frame, verbose=False, conf=0.25, iou=0.5)

    if len(inference_results[0].boxes) > 0:
        # Formato estricto requerido por BoxMOT: [x_min, y_min, x_max, y_max, confidence, class_id]
        formatted_detections = np.hstack((
            inference_results[0].boxes.xyxy.cpu().numpy(),
            inference_results[0].boxes.conf.cpu().numpy().reshape(-1, 1),
            inference_results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
        ))

        # Protección frente a fallos puntuales internos del tracker
        try:
            return tracker.update(formatted_detections, frame)
        except Exception as e:
            print(f" [!] Aviso: Salto de frame por error en tracker ({e})")
            return []

    # Si no hay abejas en este frame, pasamos un array vacío al tracker para que lo sepa
    try:
        return tracker.update(np.empty((0, 6)), frame)
    except Exception:
        return []

def run_benchmark(model_path: str, input_path: str, output_root: str) -> None:
    """
    Ejecuta un benchmark comparativo de los 5 algoritmos SOTA de tracking
    y exporta las trayectorias en formato científico MOT16.
    """
    # Tus 5 trackers obligatorios
    trackers_to_test = ['bytetrack', 'botsort', 'ocsort', 'deepocsort', 'strongsort']

    print(f"[*] Cargando detector base YOLO: {model_path}")
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encuentra el modelo en: {model_path}")
        return
        
    model = YOLO(model_path)

    if not os.path.exists(input_path):
        print(f"[ERROR] Directorio de entrada no existe: {input_path}")
        return
        
    input_items = sorted([d for d in os.listdir(input_path) if not d.startswith('.')])
    
    if not input_items:
        print(f"[!] Directorio de entrada vacío: {input_path}")
        return

    # Selección automática del mejor hardware disponible en el PC
    if torch.cuda.is_available():
        target_device = 'cuda:0'
        use_half_precision = True
        print("[*] Hardware detectado: NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        target_device = 'mps'
        use_half_precision = False
        print("[*] Hardware detectado: Apple Silicon (M1/M2/M3)")
    else:
        target_device = 'cpu'
        use_half_precision = False
        print("[*] Hardware detectado: CPU Standard (Irá más lento)")

    for tracker_type in trackers_to_test:
        print(f"\n=======================================================")
        print(f"🚀 EJECUTANDO PIPELINE DE SEGUIMIENTO: {tracker_type.upper()}")
        print(f"=======================================================")

        tracker_dest_dir = os.path.join(output_root, tracker_type)
        os.makedirs(tracker_dest_dir, exist_ok=True)

        for item_name in input_items:
            item_source_path = os.path.join(input_path, item_name)
            print(f"  🎬 Procesando secuencia: {item_name}")

            # Inicialización del tracker. 
            # Descargará automáticamente 'osnet_x0_25_msmt17.pt' la primera vez.
            tracker = create_tracker(
                tracker_type=tracker_type,
                tracker_config=None,
                reid_weights="osnet_x0_25_msmt17.pt",
                device=target_device,
                half=use_half_precision
            )

            mot_formatted_trajectories = []

            # Flujo para procesar un vídeo (.mp4, .avi, etc)
            if os.path.isfile(item_source_path):
                video_capture = cv2.VideoCapture(item_source_path)
                frame_number = 1

                while True:
                    success, frame = video_capture.read()
                    if not success:
                        break

                    active_tracks = realizar_tracking(frame, model, tracker)

                    for track in active_tracks:
                        # BoxMOT devuelve array con formato: [x1, y1, x2, y2, track_id, conf, class, ind]
                        x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], int(track[4])
                        
                        # Limpieza matemática: evitamos píxeles negativos que rompen los evaluadores
                        x1, y1 = max(0, float(x1)), max(0, float(y1))
                        width, height = max(1, float(x2 - x1)), max(1, float(y2 - y1))

                        # Formato MOT16: frame, id, x, y, width, height, conf, -1, -1, -1
                        mot_formatted_trajectories.append(
                            f"{frame_number},{track_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},1,-1,-1,-1"
                        )

                    frame_number += 1

                video_capture.release()

                # Guardamos el archivo .txt con el mismo nombre que el vídeo
                output_file_basename = os.path.splitext(item_name)[0]
                results_dest_path = os.path.join(tracker_dest_dir, f"{output_file_basename}.txt")

                with open(results_dest_path, "w") as f:
                    f.write("\n".join(mot_formatted_trajectories))

                print(f"    ✅ Trazas exportadas: {results_dest_path}")

    print(f"\n🏆 BENCHMARK FINALIZADO. Resultados almacenados en: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecución de Benchmark para algoritmos SOTA de Multiple Object Tracking (TFG)"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta a tu modelo entrenado de YOLO (ej: best.pt)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Carpeta que contiene los vídeos a evaluar (.mp4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_resultados",
        help="Carpeta donde se guardarán los .txt (por defecto: 'benchmark_resultados')"
    )

    args = parser.parse_args()
    run_benchmark(args.model, args.input, args.output)