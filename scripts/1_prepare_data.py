import shutil
import argparse
from pathlib import Path

def copy_file(src: Path, dst: Path):
    """
    Función auxiliar para la copia segura de archivos.
    Crea la jerarquía de directorios necesaria y evita redundancias en re-ejecuciones.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Eficiencia: Solo realizamos la operación de I/O si el archivo no existe en destino
    if not dst.exists():
        shutil.copy2(src, dst)

def process_split(split_name: str, split_seqs: list, det_root: Path, out_dir: Path):
    """
    Procesa un subconjunto de datos (train, val o test).
    Realiza la extracción, el renombrado para evitar colisiones espaciales y 
    gestiona la generación de muestras negativas para el entrenamiento.
    """
    img_out = out_dir / split_name / "images"
    lbl_out = out_dir / split_name / "labels"
    
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    n_img, n_lbl = 0, 0

    for seq in split_seqs:
        seq_dir = det_root / seq
        imgs_dir = seq_dir / "images"
        lbls_dir = seq_dir / "labels"

        if not imgs_dir.exists():
            continue

        for img in imgs_dir.iterdir():
            if (not img.is_file()) or (img.suffix.lower() not in IMG_EXTS):
                continue

            # [ESTRATEGIA TÉCNICA 1]: Resolución de Colisiones de Nomenclatura
            # Al unificar múltiples secuencias de vídeo en un único directorio ('train'),
            # inyectamos el ID de la secuencia al nombre del frame para garantizar unicidad.
            new_stem = f"{seq}__{img.stem}"
            img_dst = img_out / f"{new_stem}{img.suffix.lower()}"
            copy_file(img, img_dst)
            n_img += 1

            lbl_src = lbls_dir / f"{img.stem}.txt"
            lbl_dst = lbl_out / f"{new_stem}.txt"
            
            if lbl_src.exists():
                copy_file(lbl_src, lbl_dst)
                n_lbl += 1
            else:
                # [ESTRATEGIA TÉCNICA 2]: Generación de Background/Negative Samples
                # Para robustecer el detector y minimizar Falsos Positivos, se generan
                # anotaciones vacías para fotogramas donde no existe la clase objetivo (abeja).
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                lbl_dst.write_text("")

    return n_img, n_lbl

def main(input_dir: str, output_dir: str):
    """
    Orquestador principal del pipeline de preparación de datos.
    Estructura el dataset bajo el estándar YOLO y genera el manifiesto data.yaml.
    """
    det_root = Path(input_dir)
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    seqs = sorted([p.name for p in det_root.iterdir() if p.is_dir()])
    print(f"[*] Total de secuencias procesadas: {len(seqs)}")

    # [ESTRATEGIA TÉCNICA 3]: Prevención de Data Leakage (Fuga de Datos)
    # Se realiza el split (train/val/test) a nivel de secuencia de vídeo y no a nivel
    # de fotograma. Esto evita que frames temporalmente adyacentes (y visualmente idénticos)
    # contaminen los conjuntos de validación/test, garantizando una evaluación objetiva.
    train_seqs = seqs[:-2]  
    val_seqs   = [seqs[-2]] 
    test_seqs  = [seqs[-1]] 

    print("[*] Generando conjunto de Entrenamiento (Train)...")
    train_imgs, train_lbls = process_split("train", train_seqs, det_root, OUT)
    
    print("[*] Generando conjunto de Validación (Val)...")
    val_imgs, val_lbls = process_split("val", val_seqs, det_root, OUT)
    
    print("[*] Generando conjunto de Evaluación (Test)...")
    test_imgs, test_lbls = process_split("test", test_seqs, det_root, OUT)

    print(f"\n[+] RESUMEN DEL DATASET:")
    print(f"    - Train: {train_imgs} imágenes / {train_lbls} etiquetas")
    print(f"    - Val:   {val_imgs} imágenes / {val_lbls} etiquetas")
    print(f"    - Test:  {test_imgs} imágenes / {test_lbls} etiquetas\n")

    # [ESTRATEGIA TÉCNICA 4]: Manifiesto YOLO
    yaml_text = (
        f"path: {OUT.absolute()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['bee']\n"
    )
    
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(yaml_text)
    print(f"[+] Manifiesto generado exitosamente en: {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de ingesta y formateo de datasets para YOLO")
    parser.add_argument("--input", type=str, default="datasets/raw/mendeley_dataset/detection",
                        help="Directorio origen con datos crudos (Raw Data)")
    parser.add_argument("--output", type=str, default="datasets/ready_for_yolo/mendeley_yolo",
                        help="Directorio destino formateado para YOLO")
    args = parser.parse_args()
    
    main(args.input, args.output)