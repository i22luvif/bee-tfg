import cv2
import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def auto_annotate_video(
    video_path: str,
    model: YOLO,  
    output_dir: str,
    frame_skip: int,
    conf_thresh: float
) -> None:
    """
    Procesa un solo vídeo y guarda sus fotogramas y etiquetas en una subcarpeta propia,
    anidada dentro de la carpeta del día correspondiente.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"[!] ERROR: No se pudo leer el vídeo {video_path}")
        return

    # Usamos Path para analizar la ruta del vídeo original
    ruta_video_obj = Path(video_path)
    video_base_name = ruta_video_obj.stem
    
    # Extraemos el nombre de la carpeta padre (el día, ej: "2026-04-20")
    carpeta_dia = ruta_video_obj.parent.name

    # ---------------------------------------------------------
    # Estructura: output_dir / DIA / NOMBRE_VIDEO / images/
    # Ejemplo: datasets/raw/mi_dataset_propio/2026-04-20/2026-04-20_12-00/images/
    # ---------------------------------------------------------
    seq_dest_dir = Path(output_dir) / carpeta_dia / video_base_name
    images_dest_dir = seq_dest_dir / "images"
    labels_dest_dir = seq_dest_dir / "labels"
    confs_dest_dir = seq_dest_dir / "confidences"

    images_dest_dir.mkdir(parents=True, exist_ok=True)
    labels_dest_dir.mkdir(parents=True, exist_ok=True)
    confs_dest_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_frames_count = 0

    print(f"  🎬 Procesando: {video_base_name} (Carpeta: {carpeta_dia}) | Extrayendo 1 de cada {frame_skip} frames...")

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        if frame_index % frame_skip == 0:
            results = model(frame, conf=conf_thresh, verbose=False)

            frame_base_name = f"{video_base_name}_{frame_index:05d}"
            image_dest_path = images_dest_dir / f"{frame_base_name}.jpg"
            label_dest_path = labels_dest_dir / f"{frame_base_name}.txt"
            conf_dest_path = confs_dest_dir / f"{frame_base_name}.txt"

            cv2.imwrite(str(image_dest_path), frame)

            with open(label_dest_path, "w") as label_file, open(conf_dest_path, "w") as conf_file:
                boxes = results[0].boxes

                if len(boxes) > 0:
                    coords = boxes.xywhn.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for cls, coord, conf in zip(classes, coords, confs):
                        linea_yolo = f"{int(cls)} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}\n"
                        label_file.write(linea_yolo)

                        linea_conf = f"{int(cls)} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f} {conf:.4f}\n"
                        conf_file.write(linea_conf)

            saved_frames_count += 1
        frame_index += 1

    video_capture.release()
    print(f"    ✅ Listo. {saved_frames_count} imágenes extraídas en {carpeta_dia}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesamiento por Lotes de Auto-etiquetado YOLO")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Carpeta raíz que contiene todos los vídeos (puede tener subcarpetas por día)"
    )
    parser.add_argument("--model", type=str, default="model/best_bee_medium.pt")
    parser.add_argument("--output", type=str, default="datasets/raw/mi_dataset_propio")
    parser.add_argument("--skip", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.4)

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    
    videos = list(input_path.rglob("*.mp4"))

    if not videos:
        print(f"[!] No se encontraron vídeos .mp4 en la carpeta {args.input_dir}")
    else:
        print("=" * 60)
        print(f"🚀 INICIANDO PROCESAMIENTO POR LOTES: {len(videos)} VÍDEOS ENCONTRADOS")
        print("=" * 60)

        # Cargamos el modelo UNA SOLA VEZ fuera del bucle para que vaya rapidísimo
        print(f"[*] Cargando modelo YOLO: {args.model}\n")
        yolo_model = YOLO(args.model)

        # Bucle mágico que lo hace todo automático
        for idx, video_file in enumerate(videos, start=1):
            print(f"[{idx}/{len(videos)}] -> {video_file.name}")
            auto_annotate_video(
                video_path=str(video_file),
                model=yolo_model,
                output_dir=args.output,
                frame_skip=args.skip,
                conf_thresh=args.conf
            )

        print("=" * 60)
        print("🎉 TODOS LOS VÍDEOS PROCESADOS CON ÉXITO")
        print("=" * 60)