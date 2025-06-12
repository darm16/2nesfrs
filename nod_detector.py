# modules/behavioral/nod_detector.py

import numpy as np
import time
from typing import Tuple, List, Optional
from logging_setup import logger

class NodDetector:
    """
    Detecta patrones de cabeceo (nods) analizando una secuencia de ángulos de cabeza.
    Es una clase con estado que mantiene un historial para identificar movimientos rápidos
    y patrones de recuperación típicos de los microsueños.
    """
    def __init__(self, config: dict = None):
        """
        Inicializa el detector de cabeceo.

        Args:
            config (dict, optional): Un diccionario con los parámetros de configuración.
                                     Si es None, se usarán valores por defecto.
        """
        if config is None:
            config = {}

        # Parámetros configurables con valores por defecto robustos
        default_config = {
            'window_size': 20,
            'min_frames_for_analysis': 10,
            'y_angle_threshold': 12.0,       # Umbral principal para el cambio en ángulo Y (grados)
            'y_recovery_threshold': 6.0,     # Umbral secundario para detectar el patrón de recuperación
            'min_velocity_dps': 100.0,       # Velocidad mínima en grados por segundo
            'max_velocity_dps': 800.0,       # Velocidad máxima para filtrar movimientos irreales
            'recovery_pattern_weight': 1.5,
            'cooldown_period_secs': 5.0,
            'consecutive_frames_threshold': 2
        }
        
        # Sobrescribir defaults con la configuración proporcionada
        self.config = {**default_config, **config}
        
        self.angle_y_history: List[float] = []
        self.timestamp_history: List[float] = []
        self.last_detection_time: float = 0
        self.risk_score: float = 0.0
        
        logger.info(f"NodDetector inicializado con la configuración: {self.config}")
        
    def reset(self):
        """Reinicia el estado interno del detector para una nueva sesión."""
        self.angle_y_history = []
        self.timestamp_history = []
        self.last_detection_time = 0
        self.risk_score = 0.0
        logger.info("Estado de NodDetector reseteado.")

    def update(self, angle_y: Optional[float]) -> Tuple[bool, float]:
        """
        Actualiza el detector con el nuevo ángulo de cabeza (eje Y, pitch).

        Args:
            angle_y (Optional[float]): El ángulo de inclinación vertical de la cabeza en grados.

        Returns:
            Tuple[bool, float]: Una tupla conteniendo (si se detectó un cabeceo, nivel de riesgo actual de 0 a 10).
        """
        if angle_y is None or not np.isfinite(angle_y):
            return False, self.risk_score

        current_time = time.time()
        
        # Añadir nuevos datos y mantener el tamaño de la ventana
        self.angle_y_history.append(angle_y)
        self.timestamp_history.append(current_time)
        
        if len(self.angle_y_history) > self.config['window_size']:
            self.angle_y_history.pop(0)
            self.timestamp_history.pop(0)
            
        # Reducir gradualmente el factor de riesgo con el tiempo
        self.risk_score = max(0, self.risk_score - 0.01)

        # Verificar si hay suficientes datos para un análisis fiable
        if len(self.angle_y_history) < self.config['min_frames_for_analysis']:
            return False, self.risk_score
            
        return self._detect_nodding()

    def _detect_nodding(self) -> Tuple[bool, float]:
        """Lógica interna para analizar el historial y detectar un patrón de cabeceo."""
        
        # Calcular diferencias entre frames consecutivos
        delta_angles = np.diff(self.angle_y_history)
        delta_times = np.diff(self.timestamp_history)
        
        # Evitar división por cero si los timestamps son idénticos
        delta_times = np.clip(delta_times, 1e-6, None)
        
        # Calcular velocidades en grados por segundo
        velocities_dps = np.abs(delta_angles / delta_times)

        # --- Criterios de Detección ---
        
        # 1. ¿Hay un movimiento descendente rápido y significativo?
        # Buscamos un cambio que supere el umbral de ángulo y velocidad
        large_dip = False
        for i in range(len(delta_angles)):
            if (delta_angles[i] > self.config['y_angle_threshold'] and # Movimiento hacia abajo (pitch positivo en muchas configs)
                velocities_dps[i] > self.config['min_velocity_dps'] and
                velocities_dps[i] < self.config['max_velocity_dps']):
                large_dip = True
                break

        # 2. ¿Hay un patrón de recuperación (movimiento opuesto)?
        # Buscamos un cambio de signo en los deltas (ej. +15 grados seguido de -10)
        recovery_pattern = any(d1 * d2 < 0 and abs(d1) > self.config['y_recovery_threshold'] and abs(d2) > self.config['y_recovery_threshold'] 
                             for d1, d2 in zip(delta_angles, delta_angles[1:]))
        
        is_nodding = False
        # Un cabeceo se considera la combinación de una caída rápida y una recuperación
        if large_dip and recovery_pattern:
            current_time = time.time()
            # Aplicar un período de enfriamiento para no generar alertas continuas
            if (current_time - self.last_detection_time) > self.config['cooldown_period_secs']:
                is_nodding = True
                self.last_detection_time = current_time
                self.risk_score += self.config['recovery_pattern_weight'] # Aumentar el riesgo
                logger.warning(f"¡Cabeceo detectado! Puntuación de riesgo aumentada a: {self.risk_score:.2f}")

        # Limitar el riesgo a un máximo de 10
        self.risk_score = min(10, self.risk_score)
        
        return is_nodding, self.risk_score

    def get_risk_level(self) -> Tuple[str, float]:
        """Devuelve una categoría de riesgo y el valor numérico."""
        risk_value = self.risk_score
        if risk_value >= 7.0:
            return "ALTO", risk_value
        elif risk_value >= 3.5:
            return "MEDIO", risk_value
        else:
            return "BAJO", risk_value