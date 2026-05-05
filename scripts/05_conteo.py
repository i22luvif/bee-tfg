import cv2
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# ── Zona de conteo: franja FINA justo encima de la piquera ───────────────────
# Ajusta y1 para subir/bajar la franja. Cuanto más estrecha, menos falsos positivos.
COUNTING_BOX = (70, 560, 1210, 635)   # (x1, y1, x2, y2)

# Margen vertical (px) dentro del box para detección anticipada cerca del borde inferior.
# Si el tracker suelta el ID a esta distancia del borde inferior → cuenta igualmente.
BOTTOM_MARGIN = 40


def point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def get_box_side(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    if point_inside_box(point, box):
        return "inside"
    if y < y1:
        return "top"      # encima del box → zona de vuelo
    if y > y2:
        return "bottom"   # debajo del box → dentro de la colmena
    if x < x1:
        return "left"
    if x > x2:
        return "right"
    return "unknown"


def near_bottom(point, box, margin=BOTTOM_MARGIN):
    """True si el punto está dentro del box pero a ≤ margin px del borde inferior."""
    x, y = point
    x1, y1, x2, y2 = box
    return point_inside_box(point, box) and (y2 - y) <= margin


def get_zone(point, box, bottom_margin):
    """
    Clasifica un punto (x,y) en una de las 3 zonas maestras.
    """
    x, y = point
    x1, y1, x2, y2 = box
    trigger_line = y2 - bottom_margin
    
    # 1. Si está fuera de la caja amarilla (por los lados o por arriba)
    if x < x1 or x > x2 or y < y1:
        return "OUTSIDE"
        
    # 2. Si está en la franja inferior (La zona crítica de la piquera)
    if y >= trigger_line:
        return "PIQUERA"
        
    # 3. Si está dentro de la caja amarilla, pero volando por encima de la piquera
    return "AIRSPACE"


def classify_crossing(history, box, bottom_margin=BOTTOM_MARGIN):
    """
    Evalúa el movimiento basándose en el origen y destino de la abeja,
    ignorando la trayectoria exacta para permitir vuelos curvos/diagonales.
    """
    # Exigimos un mínimo de 3 frames de vida para ignorar detecciones fantasma
    if len(history) < 3:
        return None

    # Mapeamos todo el historial de la abeja a Zonas
    zones = [get_zone(h[1], box, bottom_margin) for h in history]
    
    first_zone = zones[0]
    last_zone = zones[-1]
    
    # Si la abeja nació y murió en la misma zona, no hizo nada útil (ej. revoloteó)
    if first_zone == last_zone:
        return None

    # ── LÓGICA IN ────────────────────────────────────────────────────────
    # Si empezó fuera o volando, y terminó desapareciendo en el agujero:
    if first_zone in ["OUTSIDE", "AIRSPACE"] and last_zone == "PIQUERA":
        return "IN"

    # ── LÓGICA OUT ───────────────────────────────────────────────────────
    # Si nació mágicamente en el agujero, y terminó desapareciendo volando o fuera:
    if first_zone == "PIQUERA" and last_zone in ["OUTSIDE", "AIRSPACE"]:
        return "OUT"

    return None


def draw_overlay(frame, box, bottom_margin=BOTTOM_MARGIN):
    x1, y1, x2, y2 = box

    # Counting box principal
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        frame, "COUNTING ZONE",
        (x1, max(y1 - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
    )

    # Franja de detección anticipada (borde inferior)
    cv2.rectangle(frame, (x1, y2 - bottom_margin), (x2, y2), (0, 165, 255), 1)
    cv2.putText(
        frame, "IN/OUT zone",
        (x1 + 4, y2 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1,
    )


def count_bees_box_method(
    video_path: str,
    model_path: str,
    output_video_path: str,
    output_csv_path: str,
    tracker_config: str = "bytetrack.yaml",
    conf: float = 0.25,
    imgsz: int = 960,
    min_track_age: int = 3,
    bottom_margin: int = BOTTOM_MARGIN,
) -> None:

    box = COUNTING_BOX
    model = YOLO(model_path)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    fps    = video_capture.get(cv2.CAP_PROP_FPS)
    width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    track_history     = defaultdict(list)
    counted_ids       = set()
    active_last_frame = set()

    in_count  = 0
    out_count = 0
    frame_index = 0

    csv_file   = open(output_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "time_seconds", "track_id", "event",
        "first_side", "last_side", "dy",
        "x_center", "y_center", "in_count", "out_count",
    ])

    def try_classify(track_id, current_frame):
        nonlocal in_count, out_count

        if track_id in counted_ids:
            return None

        history  = track_history.get(track_id, [])
        relevant = [h for h in history if h[2] != "unknown"]

        if len(relevant) < min_track_age:
            return None

        event = classify_crossing(relevant, box, bottom_margin)
        if event is None:
            return None

        counted_ids.add(track_id)
        if event == "IN":
            in_count += 1
        else:
            out_count += 1

        pts        = [h[1] for h in relevant]
        dy         = pts[-1][1] - pts[0][1]
        last_pt    = relevant[-1][1]

        csv_writer.writerow([
            current_frame,
            round(current_frame / fps, 3) if fps > 0 else 0,
            track_id, event,
            relevant[0][2], relevant[-1][2], round(dy, 1),
            last_pt[0], last_pt[1],
            in_count, out_count,
        ])

        return event, last_pt

    results_stream = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker_config,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    for result in results_stream:
        frame       = result.orig_img.copy()
        frame_index += 1

        draw_overlay(frame, box, bottom_margin)

        active_this_frame = set()
        boxes_result = result.boxes

        if boxes_result is not None and boxes_result.id is not None:
            track_ids   = boxes_result.id.int().cpu().tolist()
            xyxy_boxes  = boxes_result.xyxy.cpu().numpy()
            confidences = boxes_result.conf.cpu().numpy()

            for track_id, bbox, det_conf in zip(track_ids, xyxy_boxes, confidences):
                x1b, y1b, x2b, y2b = bbox
                x_center = int((x1b + x2b) / 2)
                y_center = int((y1b + y2b) / 2)

                current_point = (x_center, y_center)
                current_side  = get_box_side(current_point, box)

                track_history[track_id].append((frame_index, current_point, current_side))
                active_this_frame.add(track_id)

                # Color según posición respecto al box
                if current_side == "bottom":
                    color = (0, 0, 255)      # rojo  → dentro colmena
                elif near_bottom(current_point, box, bottom_margin):
                    color = (0, 165, 255)    # naranja → zona crítica
                elif current_side == "inside":
                    color = (0, 255, 0)      # verde → en la franja
                else:
                    color = (160, 160, 160)  # gris  → fuera del box

                cv2.rectangle(frame, (int(x1b), int(y1b)), (int(x2b), int(y2b)), color, 2)
                cv2.circle(frame, current_point, 4, (255, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"ID {track_id} {det_conf:.2f} [{current_side}]",
                    (int(x1b), max(int(y1b) - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        # ── Tracks que desaparecieron este frame ──────────────────────
        for track_id in (active_last_frame - active_this_frame):
            result_ev = try_classify(track_id, frame_index)
            if result_ev is not None:
                event, last_pt = result_ev
                color_ev = (255, 200, 0) if event == "IN" else (0, 100, 255)
                cv2.putText(
                    frame, f">>> {event} ID {track_id}",
                    (last_pt[0] + 10, last_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ev, 2,
                )

        active_last_frame = active_this_frame

        # ── HUD ───────────────────────────────────────────────────────
        cv2.rectangle(frame, (10, 10), (310, 105), (0, 0, 0), -1)
        cv2.putText(frame, f"IN:  {in_count}",  (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
        cv2.putText(frame, f"OUT: {out_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 180, 255), 2)

        video_writer.write(frame)

    # ── Tracks aún activos al terminar el vídeo ───────────────────────
    for track_id in active_last_frame:
        try_classify(track_id, frame_index)

    csv_file.close()
    video_capture.release()
    video_writer.release()

    print("[+] Conteo finalizado")
    print(f"    IN:  {in_count}")
    print(f"    OUT: {out_count}")
    print(f"    Vídeo guardado en: {output_video_path}")
    print(f"    CSV guardado en:   {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conteo de abejas — piquera en borde inferior del frame"
    )
    parser.add_argument("--video",          type=str,   required=True)
    parser.add_argument("--model",          type=str,   required=True)
    parser.add_argument("--output_video",   type=str,   default="runs/counting/counting_output.mp4")
    parser.add_argument("--output_csv",     type=str,   default="runs/counting/counting_events.csv")
    parser.add_argument("--tracker",        type=str,   default="bytetrack.yaml")
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--imgsz",          type=int,   default=960)
    parser.add_argument("--min_track_age",  type=int,   default=3,
                        help="Frames mínimos de track para considerarlo válido (default: 3)")
    parser.add_argument("--bottom_margin",  type=int,   default=BOTTOM_MARGIN,
                        help="Px desde el borde inferior del box para detección anticipada (default: 40)")

    args = parser.parse_args()

    count_bees_box_method(
        video_path=args.video,
        model_path=args.model,
        output_video_path=args.output_video,
        output_csv_path=args.output_csv,
        tracker_config=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        min_track_age=args.min_track_age,
        bottom_margin=args.bottom_margin,
    )