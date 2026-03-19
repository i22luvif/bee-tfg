import shutil
import argparse
from pathlib import Path

def copy_file(src: Path, dst: Path):
    """
    Función auxiliar para copiar archivos de forma segura.
    Crea las carpetas de destino si no existen y evita copiar si el archivo ya está allí.
    """
    # Crea toda la ruta de carpetas padre si no existen (equivale a mkdir -p en Linux)
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Solo copia el archivo si no existe ya en el destino, ahorrando tiempo en re-ejecuciones
    if not dst.exists():
        shutil.copy2(src, dst)

def process_split(split_name, split_seqs, det_root, out_dir):
    """
    Procesa un conjunto de datos (train, val o test).
    Busca las imágenes y etiquetas de las secuencias indicadas, las renombra para evitar
    colisiones y las copia a la carpeta final de YOLO.
    """
    # Definimos las rutas de destino para este split (ej: datasets/.../train/images)
    img_out = out_dir / split_name / "images"
    lbl_out = out_dir / split_name / "labels"
    
    # Extensiones de imagen válidas soportadas por YOLO
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    n_img, n_lbl = 0, 0

    # Iteramos sobre cada carpeta de secuencia asignada a este split (ej: SEQ_01, SEQ_02)
    for seq in split_seqs:
        seq_dir = det_root / seq
        imgs_dir = seq_dir / "images"
        lbls_dir = seq_dir / "labels"

        # Si la secuencia no tiene carpeta de imágenes, la saltamos
        if not imgs_dir.exists():
            continue

        # Procesamos cada archivo dentro de la carpeta de imágenes de la secuencia
        for img in imgs_dir.iterdir():
            # Filtramos para asegurarnos de que es un archivo y tiene extensión de imagen
            if (not img.is_file()) or (img.suffix.lower() not in IMG_EXTS):
                continue

            # --- TRUCO CRÍTICO: RENOMBRADO PARA EVITAR COLISIONES ---
            # Como SEQ_01 y SEQ_02 pueden tener ambos una foto llamada "0001.jpg", 
            # le inyectamos el nombre de la secuencia al principio: "SEQ_01__0001.jpg".
            # Así, al juntar todo en la misma carpeta 'train', no se sobrescriben.
            new_stem = f"{seq}__{img.stem}"
            img_dst = img_out / f"{new_stem}{img.suffix.lower()}"
            copy_file(img, img_dst)
            n_img += 1

            # Buscamos si existe la etiqueta (.txt) correspondiente a esa imagen
            lbl_src = lbls_dir / f"{img.stem}.txt"
            lbl_dst = lbl_out / f"{new_stem}.txt"
            
            if lbl_src.exists():
                # Si existe, la copiamos con el nuevo nombre
                copy_file(lbl_src, lbl_dst)
                n_lbl += 1
            else:
                # --- TRUCO AVANZADO: NEGATIVE SAMPLES ---
                # Si la imagen no tiene etiqueta (no hay abejas), YOLO requiere un .txt vacío.
                # Esto le enseña a la red a reconocer el "fondo" y reduce los falsos positivos.
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                lbl_dst.write_text("")

    return n_img, n_lbl

def main(input_dir, output_dir):
    """
    Función principal que orquesta la división del dataset y la creación del data.yaml.
    """
    det_root = Path(input_dir)
    OUT = Path(output_dir)
    
    # Creamos la carpeta raíz de salida
    OUT.mkdir(parents=True, exist_ok=True)

    # Obtenemos la lista de todas las carpetas de secuencias ordenadas alfabéticamente
    seqs = sorted([p.name for p in det_root.iterdir() if p.is_dir()])
    print(f"[*] Nº secuencias encontradas: {len(seqs)}")

    # --- PREVENCIÓN DE DATA LEAKAGE (Fuga de Datos) ---
    # Dividimos por secuencias de vídeo enteras en lugar de imágenes sueltas.
    # Si dividiéramos imágenes aleatorias, fotos casi idénticas acabarían en train y test,
    # falseando las métricas porque el modelo "memorizaría" en lugar de "aprender".
    train_seqs = seqs[:-2]  # Todas las secuencias menos las dos últimas
    val_seqs   = [seqs[-2]] # La penúltima secuencia
    test_seqs  = [seqs[-1]] # La última secuencia

    print("[*] Procesando Train...")
    tr = process_split("train", train_seqs, det_root, OUT)
    
    print("[*] Procesando Val...")
    va = process_split("val", val_seqs, det_root, OUT)
    
    print("[*] Procesando Test...")
    te = process_split("test", test_seqs, det_root, OUT)

    print(f"[+] Resumen -> Train: {tr} | Val: {va} | Test: {te}")

    # --- GENERACIÓN DEL ARCHIVO DE CONFIGURACIÓN DE YOLO ---
    # Este YAML es el "mapa" que YOLO lee para saber dónde están los datos y qué clases hay.
    yaml_text = (
        f"path: {OUT.absolute()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"                # nc = number of classes (solo 1 clase)
        "names: ['bee']\n"       # El nombre de la clase
    )
    
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(yaml_text)
    print(f"[+] Archivo data.yaml generado en: {yaml_path}")

# Punto de entrada del script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara el dataset Mendeley para YOLO")
    
    # INPUT: Lee del almacén intocable (datos crudos)
    parser.add_argument("--input", type=str, default="datasets/raw/mendeley_dataset/detection",
                        help="Ruta a las secuencias originales de Mendeley")
    
    # OUTPUT: Guarda en tu nuevo almacén de datasets limpios para YOLO
    parser.add_argument("--output", type=str, default="datasets/ready_for_yolo/mendeley_yolo",
                        help="Ruta donde se generará la estructura final para YOLO")
    
    args = parser.parse_args()
    main(args.input, args.output)