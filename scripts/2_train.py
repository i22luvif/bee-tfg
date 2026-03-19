import argparse
import os
import shutil
import torch  # Necesario para que Python se comunique con la tarjeta gráfica
from ultralytics import YOLO

def train_model(data_yaml, weights, epochs, batch, imgsz, project, name):
    """
    Orquesta el entrenamiento de la red neuronal YOLO.
    Configura el hardware automáticamente, aplica técnicas de aumento de datos
    específicas para vista cenital, y extrae el modelo óptimo al finalizar.
    """
    
    # --- PILOTO AUTOMÁTICO: DETECCIÓN DE HARDWARE ---
    # Preguntamos si el sistema tiene una tarjeta NVIDIA (GPU) instalada y lista.
    # Esto permite que el mismo código funcione en tu portátil (lento) y en Kaggle (rápido).
    if torch.cuda.is_available():
        dispositivo = 0  # YOLO usa el entero '0' para referirse a la primera GPU
        nombre_gpu = torch.cuda.get_device_name(0)
        print(f"\n[🚀] ¡Modo Turbo Activado! Entrenando en GPU: {nombre_gpu}")
    else:
        dispositivo = 'cpu'
        print("\n[🐢] No se detectó GPU compatible con CUDA. Entrenando en CPU...")
    # ------------------------------------------------

    print(f"[*] Iniciando entrenamiento con el modelo base (Transfer Learning): {weights}")
    
    # Verificación de seguridad: asegurarnos de que el dataset ha sido preparado
    if not os.path.exists(data_yaml):
        print(f"[!] ERROR: No se encuentra el archivo {data_yaml}. Ejecuta prepare_data.py primero.")
        return

    # Cargamos los pesos pre-entrenados. Es mejor partir de un modelo que ya sabe 
    # distinguir formas básicas que entrenar desde cero.
    model = YOLO(weights) 

    # --- NÚCLEO DEL ENTRENAMIENTO ---
    results = model.train(
        data=data_yaml,      # Ruta al archivo YAML generado en el paso anterior
        epochs=epochs,       # Límite máximo de iteraciones sobre todo el dataset  
        
        # EARLY STOPPING (Paciencia): 
        # Si pasan 25 épocas sin que la precisión mejore, el entrenamiento se detiene.
        # Esto evita el 'overfitting' (que el modelo memorice en lugar de aprender).
        patience=25,         
        
        imgsz=imgsz,         # Alta resolución (960) crucial para detectar insectos diminutos
        batch=batch,         # Imágenes procesadas a la vez. Reducir si da 'Out of Memory'
        device=dispositivo,  # Le pasamos la variable automática (GPU o CPU)
        project=project,     # Carpeta principal de guardado (runs/train)
        name=name,           # Nombre del experimento (creará una subcarpeta con este nombre)
        
        # --- DATA AUGMENTATION (Aumento de Datos Biológico/Físico) ---
        # Rotación 180º: Como las abejas se graban desde arriba (vista cenital), no hay "arriba" 
        # ni "abajo". Obliga a YOLO a reconocerlas sin importar hacia dónde vuelen.
        degrees=180.0,       
        
        # Mosaic y Mixup: Técnicas que juntan varias fotos en una. 
        # Son fundamentales para enseñar al modelo a detectar abejas muy agrupadas (oclusiones).
        mosaic=1.0,          
        mixup=0.2,           
        
        # Variaciones de luz y saturación (por si el sol cambia durante la grabación real)
        hsv_s=0.3,           
        hsv_v=0.3,           
        
        plots=True,          # Genera gráficas de rendimiento (útiles para la memoria del TFG)
        exist_ok=True        # Sobrescribe la carpeta del experimento si ya existe
    )
    
    # --- EXTRACCIÓN DEL "TESORO" (Mejor Modelo) ---
    # Ultralytics guarda el mejor resultado muy profundo en sus carpetas.
    # Vamos a sacarlo de ahí y ponerlo en la raíz del proyecto para mayor comodidad.
    best_model_path = os.path.join(project, name, "weights", "best.pt")
    final_dest = os.path.join("model", f"best_{name}.pt")
    
    os.makedirs("model", exist_ok=True) # Crea la carpeta 'model/' si no existe
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_dest)
        print(f"\n[+] Entrenamiento finalizado. Mejor modelo extraído y guardado en: {final_dest}")
    else:
        print("\n[!] El entrenamiento terminó pero no se encontró el archivo best.pt en la ruta esperada.")

# Punto de entrada desde la terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena YOLO para detección de abejas")
    
    # Parámetros que puedes cambiar desde la consola sin tocar el código
    parser.add_argument("--data", type=str, default="datasets/ready_for_yolo/mendeley_yolo/data.yaml")
    parser.add_argument("--weights", type=str, default="yolo26m.pt", help="Modelo base de YOLO") 
    parser.add_argument("--epochs", type=int, default=150, help="Nº máximo de iteraciones")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño del lote (bajar a 4 si falta RAM)")
    parser.add_argument("--imgsz", type=int, default=960, help="Resolución de las imágenes")
    parser.add_argument("--project", type=str, default="runs/train", help="Directorio raíz de guardado")
    parser.add_argument("--name", type=str, default="bee_mendeley_medium2", help="Nombre del experimento")
    
    args = parser.parse_args()
    
    # Lanzamos la función con los argumentos capturados
    train_model(args.data, args.weights, args.epochs, args.batch, args.imgsz, args.project, args.name)