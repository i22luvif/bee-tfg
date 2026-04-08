import argparse
import os
import shutil
import torch
from ultralytics import YOLO

def train_model(data_yaml: str, weights: str, epochs: int, batch: int, imgsz: int, project: str, name: str) -> None:
    """
    Orquesta el entrenamiento del modelo detector YOLO.
    Configura el hardware dinámicamente, aplica técnicas de aumento de datos espaciales 
    específicas para biometría cenital, y extrae los pesos óptimos al finalizar.
    """
    
    # [ESTRATEGIA TÉCNICA 1]: Asignación Dinámica de Aceleración Hardware
    # Permite la ejecución agnóstica del script (funciona en entorno local o en la nube)
    # comprobando la disponibilidad de núcleos CUDA.
    if torch.cuda.is_available():
        dispositivo = 0  # YOLO utiliza '0' para mapear la primera GPU (ej. NVIDIA Tesla T4)
        nombre_gpu = torch.cuda.get_device_name(0)
        print(f"\n[🚀] Hardware de aceleración detectado. Entrenando en GPU: {nombre_gpu}")
    else:
        dispositivo = 'cpu'
        print("\n[🐢] No se detectó GPU compatible con CUDA. Ejecutando en CPU...")

    print(f"[*] Iniciando Transfer Learning desde los pesos base: {weights}")
    
    # Validación previa al entrenamiento
    if not os.path.exists(data_yaml):
        print(f"[!] ERROR CRÍTICO: No se localiza el manifiesto {data_yaml}. Ejecute el pipeline de preparación primero.")
        return

    # Inicialización de la arquitectura YOLO con los pesos pre-entrenados
    model = YOLO(weights) 

    # [ESTRATEGIA TÉCNICA 2]: Configuración Hiperparamétrica del Entrenamiento
    results = model.train(
        data=data_yaml,      # Manifiesto del dataset
        epochs=epochs,       # Límite superior de épocas de entrenamiento
        
        # Early Stopping: Detiene la ejecución si el mAP50-95 no mejora en 25 épocas consecutivas.
        # Mecanismo crucial para mitigar el sobreajuste (Overfitting).
        patience=25,         
        
        imgsz=imgsz,         # Resolución de entrada (960px). Vital para la detección de objetos pequeños (Small Object Detection).
        batch=batch,         # Tamaño del lote (Batch Size). Ajustable según VRAM disponible.
        device=dispositivo,  
        project=project,     
        name=name,           
        
        # [ESTRATEGIA TÉCNICA 3]: Data Augmentation Específico para Biometría
        # degrees=180.0: Fundamental para capturas cenitales (desde arriba) de la piquera, 
        # ya que los insectos no tienen una orientación espacial predefinida ("arriba" o "abajo").
        degrees=180.0,       
        
        # Mosaic & Mixup: Técnicas de composición espacial que fuerzan a la red a detectar 
        # abejas en entornos de alta densidad espacial y oclusiones severas.
        mosaic=1.0,          
        mixup=0.2,           
        
        # Perturbaciones radiométricas para simular cambios de exposición solar
        hsv_s=0.3,           
        hsv_v=0.3,           
        
        plots=True,          # Genera automáticamente las curvas de Loss y Precisión/Recall
        exist_ok=True        # Evita la detención del script si el directorio ya existe
    )
    
    # [ESTRATEGIA TÉCNICA 4]: Extracción Automática de Pesos Óptimos
    # El framework almacena múltiples checkpoints. Este bloque aísla exclusivamente 
    # el archivo 'best.pt' (modelo con menor función de pérdida en el conjunto de validación)
    # y lo sitúa en la raíz del proyecto para facilitar la inferencia posterior.
    best_model_path = os.path.join(project, name, "weights", "best.pt")
    final_dest = os.path.join("model", f"best_{name}.pt")
    
    os.makedirs("model", exist_ok=True) 
    
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_dest)
        print(f"\n[+] Pipeline finalizado. Mejores pesos extraídos en: {final_dest}")
    else:
        print("\n[!] Advertencia: El entrenamiento concluyó, pero no se generó el checkpoint best.pt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento YOLO - Detección de Apis Mellifera")
    
    parser.add_argument("--data", type=str, default="datasets/ready_for_yolo/mendeley_yolo/data.yaml")
    parser.add_argument("--weights", type=str, default="yolo26m.pt", help="Pesos base para Transfer Learning") 
    parser.add_argument("--epochs", type=int, default=150, help="Iteraciones máximas")
    parser.add_argument("--batch", type=int, default=8, help="Imágenes por iteración (reducir si hay error OOM)")
    parser.add_argument("--imgsz", type=int, default=960, help="Dimensión tensorial de entrada")
    parser.add_argument("--project", type=str, default="runs/train", help="Directorio de logs/métricas")
    parser.add_argument("--name", type=str, default="bee_model", help="Nomenclatura del experimento")
    
    args = parser.parse_args()
    
    train_model(args.data, args.weights, args.epochs, args.batch, args.imgsz, args.project, args.name)