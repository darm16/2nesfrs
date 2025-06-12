# pose_estimator.py

import cv2
import math
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from logging_setup import logger

class PoseEstimator:
    """
    Clase estática que contiene la lógica para estimar la pose de la cabeza
    a partir de landmarks faciales y determinar si es aceptable.
    """

    # Índices de landmarks de MediaPipe usados para la estimación de pose.
    # Son puntos estables en el rostro: punta de la nariz, barbilla, esquinas de ojos y boca.
    LANDMARK_INDICES_FOR_POSE = [1, 33, 263, 61, 291, 199]

    @staticmethod
    def estimate_head_pose(frame_shape: Tuple[int, int], face_landmarks: Any) -> Optional[Tuple[float, float, float]]:
        """
        Estima los ángulos de pose de cabeza (Pitch, Yaw, Roll) a partir de los landmarks faciales.

        Args:
            frame_shape (Tuple[int, int]): La forma del frame (alto, ancho).
            face_landmarks (Any): El objeto de landmarks faciales de MediaPipe.

        Returns:
            Optional[Tuple[float, float, float]]: Tupla con (Pitch, Yaw, Roll) en grados, o None si falla.
        """
        h, w = frame_shape
        landmarks_list = face_landmarks.landmark

        try:
            # Puntos 2D de la imagen, extraídos de los landmarks detectados
            image_points = np.array([
                (landmarks_list[i].x * w, landmarks_list[i].y * h) for i in PoseEstimator.LANDMARK_INDICES_FOR_POSE
            ], dtype="double")

            # Puntos 3D de un modelo genérico de cabeza. Estos son estándar y no cambian.
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Punta de la nariz
                (-225.0, 170.0, -135.0),     # Comisura del ojo izquierdo
                (225.0, 170.0, -135.0),      # Comisura del ojo derecho
                (-150.0, -150.0, -125.0),    # Comisura de la boca izquierda
                (150.0, -150.0, -125.0),     # Comisura de la boca derecha
                (0.0, -330.0, -65.0)         # Punta de la barbilla
            ])

            # Parámetros de la cámara (asunciones estándar)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            # Asumimos que no hay distorsión de lente
            dist_coeffs = np.zeros((4, 1))

            # Resolver el problema Perspective-n-Point (PnP) para obtener la rotación y traslación
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                logger.warning("cv2.solvePnP no pudo estimar la pose de la cabeza.")
                return None

            # Convertir el vector de rotación a una matriz de rotación
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Proyectar un eje 3D para visualizar la pose (útil para depuración)
            # (Esta parte se puede omitir si no se necesita visualización)
            
            # Calcular los ángulos de Euler (pitch, yaw, roll) a partir de la matriz de rotación
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = math.atan2(-rotation_matrix[2, 0], sy)
                z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = math.atan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            # Convertir de radianes a grados
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)
            
            return pitch, yaw, roll

        except Exception as e:
            logger.error(f"Error inesperado durante la estimación de pose: {e}", exc_info=True)
            return None

    @staticmethod
    def is_pose_acceptable(pitch: float, yaw: float, roll: float, config: Dict) -> bool:
        """
        Verifica si los ángulos de pose están dentro de los límites aceptables definidos en la configuración.

        Args:
            pitch (float): Ángulo de Pitch en grados.
            yaw (float): Ángulo de Yaw en grados.
            roll (float): Ángulo de Roll en grados.
            config (Dict): El diccionario de configuración de la aplicación.

        Returns:
            bool: True si la pose es aceptable, False en caso contrario.
        """
        pose_limits = config.get('recognition_settings', {}).get('pose_limits', {})
        max_pitch = pose_limits.get('MAX_ABS_PITCH', 20.0)
        max_yaw = pose_limits.get('MAX_ABS_YAW', 25.0)
        max_roll = pose_limits.get('MAX_ABS_ROLL', 20.0)

        return (abs(pitch) <= max_pitch and
                abs(yaw) <= max_yaw and
                abs(roll) <= max_roll)