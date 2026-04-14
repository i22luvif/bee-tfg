import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


def auto_annotate_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    frame_skip: int,
    conf_thresh: float
) -> None:
    """
    Motor de auto-etiquetado (pseudo-labeling) para vídeos.

    Procedimiento:
    1. Carga un vídeo de entrada.
    2. Extrae fotogramas de forma espaciada mediante frame skipping.
    3. Aplica un modelo YOLO sobre cada fotograma seleccionado.
    4. Guarda:
       - la imagen del fotograma,
       - un archivo de etiquetas en formato YOLO estándar,
       - y un archivo auxiliar con las mismas etiquetas más la confianza.

    Parámetros:
        video_path (str): ruta al vídeo de entrada.
        model_path (str): ruta al modelo YOLO que se utilizará como asistente.
        output_dir (str): directorio raíz donde se guardarán los resultados.
        frame_skip (int): procesa 1 de cada N fotogramas.
        conf_thresh (float): umbral mínimo de confianza de las detecciones.
    """

    print(f"[*] Cargando modelo asistente YOLO: {model_path}")
    model = YOLO(model_path)

    # Apertura del vídeo de entrada.
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"[!] ERROR: No se pudo leer el vídeo {video_path}")
        return

    # Estructura de salida compatible con pipelines basados en YOLO.
    images_dest_dir = Path(output_dir) / "images"
    labels_dest_dir = Path(output_dir) / "labels"
    confs_dest_dir = Path(output_dir) / "confidences"

    images_dest_dir.mkdir(parents=True, exist_ok=True)
    labels_dest_dir.mkdir(parents=True, exist_ok=True)
    confs_dest_dir.mkdir(parents=True, exist_ok=True)

    # Nombre base del vídeo sin extensión.
    video_base_name = Path(video_path).stem

    frame_index = 0
    saved_frames_count = 0

    print(
        f"[*] Procesando vídeo: {video_base_name} | "
        f"Extrayendo 1 de cada {frame_skip} fotogramas..."
    )

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break  # Fin del vídeo

        # Muestreo temporal: solo se procesa 1 de cada N fotogramas.
        if frame_index % frame_skip == 0:

            # Umbral de confianza bajo para priorizar cobertura en pseudo-labeling.
            results = model(frame, conf=conf_thresh, verbose=False)

            # Identificador único para la imagen y sus archivos asociados.
            frame_base_name = f"{video_base_name}_{frame_index:05d}"
            image_dest_path = images_dest_dir / f"{frame_base_name}.jpg"
            label_dest_path = labels_dest_dir / f"{frame_base_name}.txt"
            conf_dest_path = confs_dest_dir / f"{frame_base_name}.txt"

            # Exportación del fotograma.
            cv2.imwrite(str(image_dest_path), frame)

            # Exportación de etiquetas estándar y etiquetas extendidas.
            with open(label_dest_path, "w") as label_file, open(conf_dest_path, "w") as conf_file:
                boxes = results[0].boxes

                if len(boxes) > 0:
                    # Coordenadas normalizadas en formato YOLO: x_center, y_center, width, height.
                    coords = boxes.xywhn.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for cls, coord, conf in zip(classes, coords, confs):
                        # Formato YOLO estándar.
                        linea_yolo = (
                            f"{int(cls)} "
                            f"{coord[0]:.6f} {coord[1]:.6f} "
                            f"{coord[2]:.6f} {coord[3]:.6f}\n"
                        )
                        label_file.write(linea_yolo)

                        # Formato auxiliar: YOLO + confianza.
                        linea_conf = (
                            f"{int(cls)} "
                            f"{coord[0]:.6f} {coord[1]:.6f} "
                            f"{coord[2]:.6f} {coord[3]:.6f} "
                            f"{conf:.4f}\n"
                        )
                        conf_file.write(linea_conf)

            saved_frames_count += 1

        frame_index += 1

    # Liberación del recurso de vídeo.
    video_capture.release()

    print("\n" + "=" * 60)
    print("✅ PRE-ANOTACIÓN COMPLETADA CON ÉXITO")
    print(f" 📁 Fotogramas y etiquetas exportados: {saved_frames_count}")
    print(f" 📍 Ruta destino: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de auto-etiquetado YOLO para generación de datasets"
    )

    # Vídeo de entrada.
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Ruta al vídeo (.mp4, .avi) de entrada"
    )

    # Modelo YOLO asistente.
    parser.add_argument(
        "--model",
        type=str,
        default="model/best_bee_medium.pt",
        help="Ruta al modelo YOLO asistente"
    )

    # Directorio raíz de salida.
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/raw/mi_dataset_propio",
        help="Directorio de salida"
    )

    # Frecuencia de muestreo temporal.
    parser.add_argument(
        "--skip",
        type=int,
        default=15,
        help="Extraer 1 de cada 'X' fotogramas"
    )

    # Umbral mínimo de confianza.
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Umbral de confianza de YOLO"
    )

    args = parser.parse_args()

    auto_annotate_video(
        args.video,
        args.model,
        args.output,
        args.skip,
        args.conf
    )