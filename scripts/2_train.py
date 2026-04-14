import argparse
import os
import shutil
import torch
from ultralytics import YOLO


def train_model(
    data_yaml: str,
    weights: str,
    epochs: int,
    batch: int,
    imgsz: int,
    project_dest_dir: str,
    experiment_name: str
) -> None:
    """
    Entrena un detector YOLO a partir de pesos preentrenados, selecciona
    automáticamente el dispositivo de cómputo y extrae el mejor checkpoint.
    """

    # Selección automática de GPU o CPU según disponibilidad.
    if torch.cuda.is_available():
        target_device = 0
        gpu_hardware_name = torch.cuda.get_device_name(0)
        print(f"\n[🚀] Entrenando en GPU: {gpu_hardware_name}")
    else:
        target_device = "cpu"
        print("\n[🐢] No se detectó GPU compatible con CUDA. Ejecutando en CPU...")

    print(f"[*] Iniciando Transfer Learning desde los pesos base: {weights}")

    # Validación previa del manifiesto del dataset.
    if not os.path.exists(data_yaml):
        print(
            f"[!] ERROR: No se localiza el manifiesto {data_yaml}. "
            f"Ejecute primero el pipeline de preparación del dataset."
        )
        return

    # Carga del modelo base.
    model = YOLO(weights)

    # Entrenamiento del detector.
    model.train(
        data=data_yaml,
        epochs=epochs,
        patience=25,
        imgsz=imgsz,
        batch=batch,
        device=target_device,
        project=project_dest_dir,
        name=experiment_name,

        # Aumentos de datos orientados a la detección de abejas en vista cenital.
        degrees=180.0,
        mosaic=1.0,
        mixup=0.2,
        hsv_s=0.3,
        hsv_v=0.3,

        plots=True,
        exist_ok=True
    )

    # Extracción automática del mejor modelo generado.
    best_weights_source_path = os.path.join(project_dest_dir, experiment_name, "weights", "best.pt")
    best_weights_dest_path = os.path.join("model", f"best_{experiment_name}.pt")

    os.makedirs("model", exist_ok=True)

    if os.path.exists(best_weights_source_path):
        shutil.copy(best_weights_source_path, best_weights_dest_path)
        print(f"\n[+] Mejores pesos extraídos en: {best_weights_dest_path}")
    else:
        print("\n[!] Advertencia: El entrenamiento terminó, pero no se generó best.pt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Entrenamiento YOLO - Detección de Apis Mellifera"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="datasets/ready_for_yolo/mendeley_yolo/data.yaml",
        help="Ruta al archivo data.yaml"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26m.pt",
        help="Pesos base para Transfer Learning"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Número máximo de épocas"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Tamaño del batch"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Resolución de entrada"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directorio de métricas y checkpoints"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="bee_model",
        help="Nombre del experimento"
    )

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project_dest_dir=args.project, 
        experiment_name=args.name      
    )