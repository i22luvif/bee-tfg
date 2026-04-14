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
    inference_results = model(frame, verbose=False, conf=0.85, iou=0.5)

    if len(inference_results[0].boxes) > 0:
        # Formato requerido por BoxMOT:
        # [x_min, y_min, x_max, y_max, confidence, class_id]
        formatted_detections = np.hstack((
            inference_results[0].boxes.xyxy.cpu().numpy(),
            inference_results[0].boxes.conf.cpu().numpy().reshape(-1, 1),
            inference_results[0].boxes.cls.cpu().numpy().reshape(-1, 1)
        ))

        # Protección frente a fallos puntuales internos del tracker.
        try:
            return tracker.update(formatted_detections, frame)
        except Exception:
            return []

    return []


def run_benchmark(model_path: str, input_path: str, output_root: str) -> None:
    """
    Ejecuta un benchmark comparativo de varios algoritmos de tracking
    y exporta las trayectorias en formato MOT16.
    """
    trackers_to_test = ['bytetrack', 'botsort', 'ocsort', 'deepocsort', 'strongsort']

    print(f"[*] Instanciando detector base YOLO: {model_path}")
    model = YOLO(model_path)

    input_items = sorted([d for d in os.listdir(input_path)])
    if not input_items:
        print(f"[!] Directorio de entrada vacío o inexistente: {input_path}")
        return

    # Selección automática de dispositivo y uso opcional de half precision.
    if torch.cuda.is_available():
        target_device = 'cuda:0'
        use_half_precision = True
    elif torch.backends.mps.is_available():
        # Soporte nativo para chips Apple M1/M2/M3 (Metal Performance Shaders)
        target_device = 'mps'
        use_half_precision = False # En Mac, FP32 suele ser más estable para el tracking
    else:
        target_device = 'cpu'
        use_half_precision = False

    for tracker_type in trackers_to_test:
        print(f"\n🚀 EJECUTANDO PIPELINE DE SEGUIMIENTO: {tracker_type.upper()}")

        tracker_dest_dir = os.path.join(output_root, tracker_type)
        os.makedirs(tracker_dest_dir, exist_ok=True)

        for item_name in input_items:
            item_source_path = os.path.join(input_path, item_name)
            print(f"  🎬 Procesando secuencia: {item_name}")

            # Inicialización del tracker con extractor ReID.
            tracker = create_tracker(
                tracker_type,
                None,
                "osnet_x0_25_msmt17.pt",
                target_device,
                half=use_half_precision
            )

            mot_formatted_trajectories = []

            # El script acepta tanto secuencias de imágenes como archivos de vídeo.
            if os.path.isdir(item_source_path):
                images_source_dir = (
                    os.path.join(item_source_path, "img1")
                    if os.path.exists(os.path.join(item_source_path, "img1"))
                    else item_source_path
                )

                image_filenames = sorted([
                    f for f in os.listdir(images_source_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])

                for frame_index, image_filename in enumerate(image_filenames):
                    frame = cv2.imread(os.path.join(images_source_dir, image_filename))
                    if frame is None:
                        continue

                    active_tracks = realizar_tracking(frame, model, tracker)

                    # Conversión de las trayectorias al formato MOT16:
                    # frame, id, x, y, width, height, ...
                    for track in active_tracks:
                        x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], int(track[4])
                        mot_formatted_trajectories.append(
                            f"{frame_index+1},{track_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1"
                        )

            else:
                video_capture = cv2.VideoCapture(item_source_path)
                frame_number = 1

                while True:
                    success, frame = video_capture.read()
                    if not success:
                        break

                    active_tracks = realizar_tracking(frame, model, tracker)

                    for track in active_tracks:
                        x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], int(track[4])
                        mot_formatted_trajectories.append(
                            f"{frame_number},{track_id},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1"
                        )

                    frame_number += 1

                video_capture.release()

            output_file_basename = os.path.splitext(item_name)[0]
            results_dest_path = os.path.join(tracker_dest_dir, f"{output_file_basename}.txt")

            with open(results_dest_path, "w") as f:
                f.write("\n".join(mot_formatted_trajectories))

            print(f"    ✅ Trazas exportadas: {results_dest_path}")

    print(f"\n✅ BENCHMARK FINALIZADO. Resultados almacenados en: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ejecución de Benchmark para algoritmos SOTA de Multiple Object Tracking"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="model/best_bee_medium.pt",
        help="Ruta al modelo YOLO detector"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/raw/BEE24/test",
        help="Directorio con secuencias/vídeos a evaluar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/benchmark_results",
        help="Directorio de volcado de resultados MOT"
    )

    args = parser.parse_args()
    run_benchmark(args.model, args.input, args.output)