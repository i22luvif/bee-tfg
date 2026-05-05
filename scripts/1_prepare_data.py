import shutil
import argparse
import random
import csv
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_file(source_path: Path, destination_path: Path) -> None:
    """
    Copia un archivo garantizando la existencia del directorio de destino
    y evitando duplicados en re-ejecuciones del pipeline.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if not destination_path.exists():
        shutil.copy2(source_path, destination_path)


def get_sequence_stats(sequence_dir: Path) -> dict:
    """
    Calcula estadísticas básicas de una secuencia:
    número de imágenes, etiquetas positivas y etiquetas vacías.
    """
    images_dir = sequence_dir / "images"
    labels_dir = sequence_dir / "labels"

    valid_image_paths = [
        file_path for file_path in images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    total_image_count = len(valid_image_paths)
    positive_label_count = 0
    background_image_count = 0

    for image_path in valid_image_paths:
        label_file_path = labels_dir / f"{image_path.stem}.txt"

        if label_file_path.exists() and label_file_path.read_text(encoding="utf-8").strip():
            positive_label_count += 1
        else:
            background_image_count += 1

    background_ratio = background_image_count / total_image_count if total_image_count > 0 else 0.0

    return {
        "sequence_name": sequence_dir.name,
        "total_images": total_image_count,
        "positive_images": positive_label_count,
        "background_images": background_image_count,
        "background_ratio": background_ratio
    }


def choose_best_split(
    sequence_metrics: dict,
    current_split_allocations: dict,
    target_image_counts: dict,
    target_global_background_ratio: float
) -> str:
    """
    Selecciona el subconjunto (train, val, test) más adecuado para una secuencia.
    """
    best_split_name = None
    lowest_error_score = float("inf")

    for split_name in ["train", "val", "test"]:
        projected_total_images = current_split_allocations[split_name]["total_images"] + sequence_metrics["total_images"]
        projected_background_images = current_split_allocations[split_name]["background_images"] + sequence_metrics["background_images"]

        size_error_penalty = abs(projected_total_images - target_image_counts[split_name]) / max(
            target_image_counts[split_name], 1
        )

        projected_background_ratio = projected_background_images / max(projected_total_images, 1)
        background_ratio_error = abs(projected_background_ratio - target_global_background_ratio)

        total_error_score = size_error_penalty + background_ratio_error

        if total_error_score < lowest_error_score:
            lowest_error_score = total_error_score
            best_split_name = split_name

    return best_split_name


import random

def split_sequences_stratified(
    all_sequences_metrics: list[dict],
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> tuple[list[str], list[str], list[str]]:
    """
    Divide las secuencias completas en train/val/test evitando data leakage
    y repartiendo las secuencias con background entre los tres subconjuntos.
    """
    random.seed(random_seed)

    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio <= 0:
        raise ValueError("Los ratios train_ratio + val_ratio deben ser menores que 1.0")

    valid_sequences_metrics = [
        metrics for metrics in all_sequences_metrics 
        if metrics["total_images"] > 0
    ]

    total_images = sum(metrics["total_images"] for metrics in valid_sequences_metrics)

    target_image_counts = {
        "train": total_images * train_ratio,
        "val": total_images * val_ratio,
        "test": total_images * test_ratio
    }

    split_allocations = {
        "train": [],
        "val": [],
        "test": []
    }

    split_image_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    # Clasificación por tipo de secuencia
    high_background_sequences = []
    mixed_sequences = []
    high_positive_sequences = []

    for sequence_metrics in valid_sequences_metrics:
        bg_ratio = sequence_metrics["background_ratio"]

        if bg_ratio >= 0.70:
            high_background_sequences.append(sequence_metrics)
        elif bg_ratio >= 0.10:
            mixed_sequences.append(sequence_metrics)
        else:
            high_positive_sequences.append(sequence_metrics)

    random.shuffle(high_background_sequences)
    random.shuffle(mixed_sequences)
    random.shuffle(high_positive_sequences)

    # Reparto equilibrado
    for group in [high_background_sequences, mixed_sequences, high_positive_sequences]:
        for sequence_metrics in group:

            best_split_name = min(
                ["train", "val", "test"],
                key=lambda split: split_image_counts[split] / max(target_image_counts[split], 1)
            )

            # 🔥 CORRECCIÓN IMPORTANTE AQUÍ
            split_allocations[best_split_name].append(sequence_metrics["sequence_name"])
            split_image_counts[best_split_name] += sequence_metrics["total_images"]

    return (
        split_allocations["train"], 
        split_allocations["val"], 
        split_allocations["test"]
    )

def process_split(
    split_name: str,
    assigned_sequence_names: list[str],
    input_dataset_path: Path,
    output_dataset_path: Path
) -> tuple[int, int, int]:
    """
    Procesa un subconjunto del dataset, copia imágenes y etiquetas,
    renombra archivos para evitar colisiones y genera muestras negativas.
    """
    output_images_dir = output_dataset_path / split_name / "images"
    output_labels_dir = output_dataset_path / split_name / "labels"

    processed_image_count = 0
    processed_label_count = 0
    generated_background_count = 0

    for sequence_name in assigned_sequence_names:
        sequence_dir = input_dataset_path / sequence_name
        sequence_images_dir = sequence_dir / "images"
        sequence_labels_dir = sequence_dir / "labels"

        if not sequence_images_dir.exists():
            print(f"[!] Advertencia: no existe carpeta images en {sequence_dir}")
            continue

        for image_file_path in sequence_images_dir.iterdir():
            if (
                not image_file_path.is_file()
                or image_file_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS
            ):
                continue

            unique_file_prefix = f"{sequence_name}__{image_file_path.stem}"

            destination_image_path = output_images_dir / f"{unique_file_prefix}{image_file_path.suffix.lower()}"
            source_label_path = sequence_labels_dir / f"{image_file_path.stem}.txt"
            destination_label_path = output_labels_dir / f"{unique_file_prefix}.txt"

            copy_file(image_file_path, destination_image_path)
            processed_image_count += 1

            if source_label_path.exists():
                copy_file(source_label_path, destination_label_path)
                processed_label_count += 1

                if not source_label_path.read_text(encoding="utf-8").strip():
                    generated_background_count += 1
            else:
                destination_label_path.parent.mkdir(parents=True, exist_ok=True)
                destination_label_path.write_text("", encoding="utf-8")
                generated_background_count += 1

    return processed_image_count, processed_label_count, generated_background_count


def save_split_summary(
    output_dataset_path: Path,
    all_sequences_metrics: list[dict],
    train_sequence_names: list[str],
    val_sequence_names: list[str],
    test_sequence_names: list[str]
) -> None:
    """
    Guarda un CSV con la asignación de cada secuencia a train, val o test.
    """
    sequence_to_split_mapping = {}

    for sequence_name in train_sequence_names:
        sequence_to_split_mapping[sequence_name] = "train"
    for sequence_name in val_sequence_names:
        sequence_to_split_mapping[sequence_name] = "val"
    for sequence_name in test_sequence_names:
        sequence_to_split_mapping[sequence_name] = "test"

    summary_csv_path = output_dataset_path / "split_summary.csv"

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "sequence",
            "split",
            "total_images",
            "positive_images",
            "background_images",
            "background_ratio"
        ])

        for sequence_metrics in sorted(all_sequences_metrics, key=lambda metrics: metrics["sequence_name"]):
            csv_writer.writerow([
                sequence_metrics["sequence_name"],
                sequence_to_split_mapping.get(sequence_metrics["sequence_name"], "unused"),
                sequence_metrics["total_images"],
                sequence_metrics["positive_images"],
                sequence_metrics["background_images"],
                round(sequence_metrics["background_ratio"], 4)
            ])

    print(f"[+] Resumen del split guardado en: {summary_csv_path}")


def print_split_details(
    split_name: str,
    assigned_sequence_names: list[str],
    metrics_by_sequence_name: dict
) -> None:
    """
    Imprime las secuencias asignadas a un split y sus estadísticas.
    """
    total_images_in_split = sum(metrics_by_sequence_name[name]["total_images"] for name in assigned_sequence_names)
    total_positives_in_split = sum(metrics_by_sequence_name[name]["positive_images"] for name in assigned_sequence_names)
    total_background_in_split = sum(metrics_by_sequence_name[name]["background_images"] for name in assigned_sequence_names)
    split_background_ratio = total_background_in_split / total_images_in_split if total_images_in_split > 0 else 0.0

    print(f"\n[{split_name.upper()}]")
    print(f"  Secuencias: {len(assigned_sequence_names)}")
    print(f"  Imágenes: {total_images_in_split}")
    print(f"  Positivas: {total_positives_in_split}")
    print(f"  Background: {total_background_in_split} ({split_background_ratio:.1%})")

    for sequence_name in assigned_sequence_names:
        metrics = metrics_by_sequence_name[sequence_name]
        print(
            f"    - {sequence_name}: "
            f"{metrics['total_images']} img | "
            f"{metrics['positive_images']} positivas | "
            f"{metrics['background_images']} background"
        )


def main(
    input_dir_path: str,
    output_dir_path: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> None:
    """
    Estructura el dataset bajo el formato YOLO, divide las secuencias en
    train/val/test mediante split estratificado y genera data.yaml.
    """
    input_dataset_path = Path(input_dir_path)
    output_dataset_path = Path(output_dir_path)
    output_dataset_path.mkdir(parents=True, exist_ok=True)

    available_sequence_directories = sorted([
        folder_path for folder_path in input_dataset_path.iterdir()
        if folder_path.is_dir()
    ])

    print(f"[*] Total de secuencias detectadas: {len(available_sequence_directories)}")

    if len(available_sequence_directories) < 3:
        raise ValueError(
            "Se necesitan al menos 3 secuencias para generar train, val y test."
        )

    all_sequences_metrics = [get_sequence_stats(seq_dir) for seq_dir in available_sequence_directories]
    metrics_by_sequence_name = {metrics["sequence_name"]: metrics for metrics in all_sequences_metrics}

    train_sequence_names, val_sequence_names, test_sequence_names = split_sequences_stratified(
        all_sequences_metrics=all_sequences_metrics,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    print("\n[*] Split automático estratificado generado:")
    print_split_details("train", train_sequence_names, metrics_by_sequence_name)
    print_split_details("val", val_sequence_names, metrics_by_sequence_name)
    print_split_details("test", test_sequence_names, metrics_by_sequence_name)

    print("\n[*] Generando conjunto de Entrenamiento (Train)...")
    train_image_count, train_label_count, train_background_count = process_split(
        split_name="train",
        assigned_sequence_names=train_sequence_names,
        input_dataset_path=input_dataset_path,
        output_dataset_path=output_dataset_path
    )

    print("[*] Generando conjunto de Validación (Val)...")
    val_image_count, val_label_count, val_background_count = process_split(
        split_name="val",
        assigned_sequence_names=val_sequence_names,
        input_dataset_path=input_dataset_path,
        output_dataset_path=output_dataset_path
    )

    print("[*] Generando conjunto de Evaluación (Test)...")
    test_image_count, test_label_count, test_background_count = process_split(
        split_name="test",
        assigned_sequence_names=test_sequence_names,
        input_dataset_path=input_dataset_path,
        output_dataset_path=output_dataset_path
    )

    print("\n[+] RESUMEN DEL DATASET:")
    print(
        f"    - Train: {train_image_count} imágenes / {train_label_count} etiquetas "
        f"/ {train_background_count} background ({train_background_count / train_image_count:.1%})"
    )
    print(
        f"    - Val:   {val_image_count} imágenes / {val_label_count} etiquetas "
        f"/ {val_background_count} background ({val_background_count / val_image_count:.1%})"
    )
    print(
        f"    - Test:  {test_image_count} imágenes / {test_label_count} etiquetas "
        f"/ {test_background_count} background ({test_background_count / test_image_count:.1%})"
    )

    yolo_yaml_content = (
        f"path: {output_dataset_path.absolute()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['bee']\n"
    )

    yaml_file_path = output_dataset_path / "data.yaml"
    yaml_file_path.write_text(yolo_yaml_content, encoding="utf-8")

    save_split_summary(
        output_dataset_path=output_dataset_path,
        all_sequences_metrics=all_sequences_metrics,
        train_sequence_names=train_sequence_names,
        val_sequence_names=val_sequence_names,
        test_sequence_names=test_sequence_names
    )

    print(f"[+] Manifiesto generado exitosamente en: {yaml_file_path}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="Pipeline de ingesta y formateo de datasets para YOLO"
    )

    cli_parser.add_argument(
        "--input",
        type=str,
        default="datasets/raw/mendeley_dataset/detection",
        help="Directorio origen con secuencias organizadas"
    )

    cli_parser.add_argument(
        "--output",
        type=str,
        default="datasets/ready_for_yolo/bee_yolo",
        help="Directorio destino formateado para YOLO"
    )

    cli_parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proporción aproximada de imágenes para entrenamiento"
    )

    cli_parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proporción aproximada de imágenes para validación"
    )

    cli_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para hacer el split reproducible"
    )

    cli_arguments = cli_parser.parse_args()

    main(
        input_dir_path=cli_arguments.input,
        output_dir_path=cli_arguments.output,
        train_ratio=cli_arguments.train_ratio,
        val_ratio=cli_arguments.val_ratio,
        random_seed=cli_arguments.seed
    )