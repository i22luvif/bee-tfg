import shutil
import argparse
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    """
    Copia un archivo garantizando la existencia del directorio de destino
    y evitando duplicados en re-ejecuciones del pipeline.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        shutil.copy2(src, dst)


def process_split(
    split_name: str,
    split_seqs: list[str],
    input_dataset_path: Path,
    output_dataset_path: Path
) -> tuple[int, int]:
    """
    Procesa un subconjunto del dataset (train, val o test), copia imágenes
    y etiquetas, renombra archivos para evitar colisiones y genera muestras
    negativas cuando no existe anotación.
    """
    images_dest_dir = output_dataset_path / split_name / "images"
    labels_dest_dir = output_dataset_path / split_name / "labels"

    valid_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_count, label_count = 0, 0

    for sequence_folder in split_seqs:
        sequence_dir = input_dataset_path / sequence_folder
        sequence_images_dir = sequence_dir / "images"
        sequence_labels_dir = sequence_dir / "labels"

        if not sequence_images_dir.exists():
            continue

        for image_file in sequence_images_dir.iterdir():
            if (not image_file.is_file()) or (
                image_file.suffix.lower() not in valid_image_extensions
            ):
                continue

            # Se añade el identificador de la secuencia para evitar
            # colisiones de nombres al unificar varias carpetas.
            unique_name = f"{sequence_folder}__{image_file.stem}"

            image_dest_path = images_dest_dir / f"{unique_name}{image_file.suffix.lower()}"
            copy_file(image_file, image_dest_path)
            image_count += 1

            label_source_path = sequence_labels_dir / f"{image_file.stem}.txt"
            label_dest_path = labels_dest_dir / f"{unique_name}.txt"

            if label_source_path.exists():
                copy_file(label_source_path, label_dest_path)
                label_count += 1
            else:
                # En YOLO, un archivo vacío representa una imagen negativa.
                label_dest_path.parent.mkdir(parents=True, exist_ok=True)
                label_dest_path.write_text("")

    return image_count, label_count


def main(input_dir: str, output_dir: str) -> None:
    """
    Estructura el dataset bajo el formato esperado por YOLO, divide las
    secuencias en train/val/test y genera el archivo data.yaml.
    """
    input_dataset_path = Path(input_dir)
    output_dataset_path = Path(output_dir)
    output_dataset_path.mkdir(parents=True, exist_ok=True)

    # Cada subdirectorio del dataset original se interpreta como una secuencia.
    all_sequences = sorted([p.name for p in input_dataset_path.iterdir() if p.is_dir()])
    print(f"[*] Total de secuencias procesadas: {len(all_sequences)}")

    # Validación mínima para garantizar un reparto train/val/test válido.
    if len(all_sequences) < 3:
        raise ValueError(
            "Se necesitan al menos 3 secuencias para generar los subconjuntos "
            "de entrenamiento, validación y prueba."
        )

    # El reparto se realiza a nivel de secuencia para evitar data leakage
    # entre entrenamiento, validación y evaluación.
    train_seqs = all_sequences[:-2]
    val_seqs = [all_sequences[-2]]
    test_seqs = [all_sequences[-1]]

    print("[*] Generando conjunto de Entrenamiento (Train)...")
    train_imgs, train_lbls = process_split(
        "train", train_seqs, input_dataset_path, output_dataset_path
    )

    print("[*] Generando conjunto de Validación (Val)...")
    val_imgs, val_lbls = process_split(
        "val", val_seqs, input_dataset_path, output_dataset_path
    )

    print("[*] Generando conjunto de Evaluación (Test)...")
    test_imgs, test_lbls = process_split(
        "test", test_seqs, input_dataset_path, output_dataset_path
    )

    print("\n[+] RESUMEN DEL DATASET:")
    print(f"    - Train: {train_imgs} imágenes / {train_lbls} etiquetas")
    print(f"    - Val:   {val_imgs} imágenes / {val_lbls} etiquetas")
    print(f"    - Test:  {test_imgs} imágenes / {test_lbls} etiquetas\n")

    # Manifiesto del dataset en formato YOLO.
    yaml_text = (
        f"path: {output_dataset_path.absolute()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['bee']\n"
    )

    yaml_path = output_dataset_path / "data.yaml"
    yaml_path.write_text(yaml_text)

    print(f"[+] Manifiesto generado exitosamente en: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de ingesta y formateo de datasets para YOLO"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="datasets/raw/mendeley_dataset/detection",
        help="Directorio origen con datos crudos"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="datasets/ready_for_yolo/mendeley_yolo",
        help="Directorio destino formateado para YOLO"
    )

    args = parser.parse_args()
    main(args.input, args.output)