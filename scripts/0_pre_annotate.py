import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

def auto_annotate_video(video_path: str, model_path: str, output_dir: str, frame_skip: int, conf_thresh: float) -> None:
    """
    Motor de Auto-Etiquetado (Pseudo-Labeling).
    Procesa un vídeo crudo, extrae fotogramas de forma espaciada (Frame Skipping) 
    y utiliza un modelo YOLO base para generar anotaciones automáticas 
    en formato de coordenadas espaciales normalizadas (YOLOv8 Format).
    """
    print(f"[*] Cargando modelo asistente YOLO: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] ERROR: No se pudo leer el vídeo {video_path}")
        return

    # [ESTRATEGIA TÉCNICA 1]: Creación de la Estructura Pura de YOLO
    # Creamos las carpetas 'images' y 'labels' que programas como Roboflow o CVAT 
    # exigen para poder importar el dataset sin errores.
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    vid_name = Path(video_path).stem
    frame_idx = 0
    saved_count = 0

    print(f"[*] Procesando vídeo: {vid_name} | Extrayendo 1 de cada {frame_skip} fotogramas...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Fin del vídeo

        # [ESTRATEGIA TÉCNICA 2]: Muestreo Temporal (Frame Skipping)
        # Solo procesamos el fotograma si cumple la condición del salto.
        # Ej: Si frame_skip=15 en un vídeo a 30FPS, extrae 2 fotos por segundo.
        if frame_idx % frame_skip == 0:
            
            # Inferencia: Usamos un umbral de confianza más bajo de lo normal (ej: 0.4).
            # Es preferible que la IA ponga falsos positivos (fáciles de borrar a mano) 
            # a que se deje abejas sin etiquetar (difíciles de dibujar de cero).
            results = model(frame, conf=conf_thresh, verbose=False)

            # Nomenclatura única: NombreDelVideo_NumeroDeFrame.jpg
            base_name = f"{vid_name}_{frame_idx:05d}"
            img_path = out_images / f"{base_name}.jpg"
            txt_path = out_labels / f"{base_name}.txt"

            # 1. Guardamos la foto física
            cv2.imwrite(str(img_path), frame)

            # 2. Guardamos las anotaciones matemáticas
            with open(txt_path, 'w') as f:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # [ESTRATEGIA TÉCNICA 3]: Extracción de Coordenadas Normalizadas (xywhn)
                    # Extraemos X_centro, Y_centro, Ancho y Alto en formato porcentual (0.0 a 1.0)
                    coords = boxes.xywhn.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()

                    for cls, coord in zip(classes, coords):
                        # Formato YOLO oficial: Clase X_centro Y_centro Ancho Alto
                        linea = f"{int(cls)} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}\n"
                        f.write(linea)
            
            saved_count += 1

        frame_idx += 1

    cap.release()
    print("\n" + "="*60)
    print(f"✅ PRE-ANOTACIÓN COMPLETADA CON ÉXITO")
    print(f" 📁 Fotogramas y etiquetas exportados: {saved_count}")
    print(f" 📍 Ruta destino: {output_dir}")
    print("="*60)
    print("💡 Siguiente paso: Sube esta carpeta a Roboflow/CVAT, revisa que las cajas ")
    print("   estén bien ajustadas, corrige los fallos, y exporta tu Dataset Definitivo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Auto-Etiquetado YOLO para generación de Datasets")
    
    # El vídeo crudo que has grabado en la colmena
    parser.add_argument("--video", type=str, required=True, help="Ruta al vídeo (.mp4, .avi) de entrada")
    
    # Tu modelo pre-entrenado actual
    parser.add_argument("--model", type=str, default="model/best_bee_medium.pt", help="Ruta al modelo YOLO asistente")
    
    # Donde se guardarán las carpetas /images y /labels
    parser.add_argument("--output", type=str, default="datasets/raw/mi_dataset_propio", help="Directorio de salida")
    
    # Salto de fotogramas: 15 significa que en un vídeo de 30fps, guarda 2 fotos por segundo.
    parser.add_argument("--skip", type=int, default=15, help="Extraer 1 de cada 'X' fotogramas")
    
    # Confianza baja a propósito (0.4) para que anote el máximo de abejas posibles
    parser.add_argument("--conf", type=float, default=0.4, help="Umbral de confianza de YOLO")
    
    args = parser.parse_args()
    auto_annotate_video(args.video, args.model, args.output, args.skip, args.conf)