import torch
import numpy as np

class LSTMClassifier:
    """
    Clasificador LSTM para detección de fatiga basado en métricas faciales.
    
    Args:
        model_path (str): Ruta al modelo LSTM pre-entrenado.
    """
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.input_data = []
        self.model.eval()  # Poner el modelo en modo evaluación
    
    def update(self, ear, mar, puc, moe):
        """
        Actualiza el estado del clasificador con nuevas métricas.
        
        Args:
            ear (float): Valor de EAR (Eye Aspect Ratio)
            mar (float): Valor de MAR (Mouth Aspect Ratio)
            puc (float): Valor de PUC (Pupil Circularity)
            moe (float): Valor de MOE (Mouth Opening Extent)
            
        Returns:
            int or None: 1 si se detecta fatiga, 0 si no, o None si no hay suficientes muestras
        """
        self.input_data.append([ear, mar, puc, moe])
        if len(self.input_data) > 20:
            self.input_data.pop(0)
        if len(self.input_data) == 20:
            return self.classify()
        return None
    
    def classify(self):
        """
        Realiza la clasificación basada en las últimas 20 muestras.
        
        Returns:
            int: 1 si se detecta fatiga, 0 en caso contrario
        """
        model_input = [
            self.input_data[:5], self.input_data[3:8],
            self.input_data[6:11], self.input_data[9:14],
            self.input_data[12:17], self.input_data[15:]
        ]
        with torch.no_grad():
            preds = torch.sigmoid(self.model(torch.FloatTensor(model_input))).gt(0.7).int()
        return int(preds.sum() >= 5)
    
    def reset(self):
        """Reinicia el buffer de datos del clasificador."""
        self.input_data = []
