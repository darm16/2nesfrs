"""
Módulo para el registro y almacenamiento de datos de monitoreo.
Incluye funciones para guardar métricas faciales e imágenes de eventos.
"""

import os
import csv
import cv2
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Crear directorios necesarios
os.makedirs("logs/data_history", exist_ok=True)
os.makedirs("logs/sleep_events", exist_ok=True)
os.makedirs("logs/hdr_images", exist_ok=True)

class DataLogger:
    """
    Clase para manejar el registro de datos de monitoreo.
    """
    
    def __init__(self, base_dir: str = "logs"):
        """
        Inicializa el gestor de registro de datos.
        
        Args:
            base_dir: Directorio base para guardar los logs
        """
        self.base_dir = Path(base_dir)
        self.csv_file = self.base_dir / "data_history" / "features_history.csv"
        self._ensure_csv_headers()
    
    def _ensure_csv_headers(self):
        """Asegura que el archivo CSV tenga los encabezados correctos."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'ear', 'mar', 'puc', 'moe', 
                    'head_angle_x', 'head_angle_y', 'event_type'
                ])
    
    def log_features(self, timestamp: float, metrics: Dict[str, float], 
                    angles: Optional[Dict[str, float]] = None, 
                    event_type: str = '') -> None:
        """
        Guarda las métricas faciales en el archivo CSV.
        
        Args:
            timestamp: Marca de tiempo en segundos desde la época
            metrics: Diccionario con las métricas (ear, mar, puc, moe)
            angles: Diccionario con los ángulos de la cabeza (x, y)
            event_type: Tipo de evento detectado (opcional)
        """
        angle_x = angles.get('x', 0.0) if angles else 0.0
        angle_y = angles.get('y', 0.0) if angles else 0.0
        
        row = [
            datetime.fromtimestamp(timestamp).isoformat(),
            metrics.get('ear', 0.0),
            metrics.get('mar', 0.0),
            metrics.get('puc', 0.0),
            metrics.get('moe', 0.0),
            angle_x,
            angle_y,
            event_type
        ]
        
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    @staticmethod
    def save_sleep_image(image: np.ndarray, user_code: str, event_type: str = 'sleep') -> str:
        """
        Guarda una imagen de un evento de sueño o fatiga.
        
        Args:
            image: Imagen a guardar (formato BGR de OpenCV)
            user_code: Código del usuario
            event_type: Tipo de evento ('sleep', 'yawn', 'distraction', etc.)
            
        Returns:
            str: Ruta al archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_code}_{event_type}_{timestamp}.png"
        filepath = Path("logs/sleep_events") / filename
        
        # Convertir a RGB si es necesario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        cv2.imwrite(str(filepath), image_rgb)
        return str(filepath)
    
    @staticmethod
    def save_hdr_image(image: np.ndarray, user_code: str) -> str:
        """
        Guarda una imagen HDR (High Dynamic Range) para análisis posterior.
        
        Args:
            image: Imagen HDR a guardar
            user_code: Código del usuario
            
        Returns:
            str: Ruta al archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_code}_hdr_{timestamp}.tiff"
        filepath = Path("logs/hdr_images") / filename
        
        # Asegurar que la imagen esté en formato flotante para HDR
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
            
        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        return str(filepath)
