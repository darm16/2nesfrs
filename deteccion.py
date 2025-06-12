# modules/behavioral/deteccion.py

import numpy as np
import math
import time
from typing import Tuple, Optional, List, Dict, Any
import mediapipe as mp
from logging_setup import logger

# --- Funciones de Cálculo Geométrico ---

def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calcula el ángulo (en grados) entre tres puntos 2D.
    El signo del ángulo indica la dirección de la rotación.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectores BA y BC
    ba = a - b
    bc = c - b
    
    # Producto punto y cruzado
    dot_product = np.dot(ba, bc)
    cross_product = np.cross(ba, bc)
    
    # Ángulo en radianes
    angle_rad = np.arctan2(cross_product, dot_product)
    
    # Convertir a grados
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_head_angle(face_landmarks: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcula los ángulos de la cabeza (pitch y yaw) a partir de landmarks faciales.
    
    Args:
        face_landmarks: El objeto de landmarks faciales de MediaPipe.

    Returns:
        Una tupla con (angle_x, angle_y) en grados, o (None, None) si falla.
    """
    if not face_landmarks:
        return None, None
    try:
        landmarks = face_landmarks.landmark
        # Puntos clave para estimar los ángulos
        p_izq = [landmarks[234].x, landmarks[234].y]
        p_der = [landmarks[454].x, landmarks[454].y]
        nariz = [landmarks[4].x, landmarks[4].y]
        frente = [landmarks[10].x, landmarks[10].y]
        barbilla = [landmarks[152].x, landmarks[152].y]
        
        # Yaw (giro horizontal)
        angle_y = calculate_angle(p_der, nariz, p_izq)
        
        # Pitch (inclinación vertical)
        angle_x = calculate_angle(barbilla, nariz, frente)
        
        return angle_x, angle_y
    except IndexError as e:
        logger.error(f"Índice de landmark fuera de rango en calculate_head_angle: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error inesperado en calculate_head_angle: {e}")
        return None, None


# --- Funciones de Detección de Eventos Específicos ---

def detect_eye_rubbing(face_landmarks: np.ndarray, hand_landmarks: Any, config: Dict) -> int:
    """
    Detecta si una o ambas manos están cerca de los ojos.

    Args:
        face_landmarks (np.ndarray): Array de landmarks faciales en coordenadas de píxeles.
        hand_landmarks (Any): Objeto de landmarks de manos de MediaPipe.
        config (Dict): Diccionario de configuración con 'camera_settings'.

    Returns:
        int: 0 (sin frotamiento), 1 (ojo izquierdo), 2 (ojo derecho), 3 (ambos).
    """
    if face_landmarks is None or hand_landmarks is None or not hand_landmarks.multi_hand_landmarks:
        return 0

    try:
        frame_width = config['camera_settings']['FRAME_WIDTH']
        frame_height = config['camera_settings']['FRAME_HEIGHT']

        # Puntos clave de los ojos y un radio de detección
        left_eye_center = np.mean(face_landmarks[[33, 160, 158, 133]], axis=0)
        right_eye_center = np.mean(face_landmarks[[263, 387, 385, 362]], axis=0)
        
        # El radio se puede basar en la distancia entre los ojos
        eye_dist = np.linalg.norm(left_eye_center - right_eye_center)
        detection_radius = eye_dist * 0.7  # Umbral de proximidad

        # Puntos clave de las manos (puntas de los dedos)
        key_hand_points = [mp.solutions.hands.HandLandmark.THUMB_TIP, 
                           mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                           mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        left_eye_rub = False
        right_eye_rub = False

        for hand_lms in hand_landmarks.multi_hand_landmarks:
            for point_idx in key_hand_points:
                hand_point = hand_lms.landmark[point_idx]
                hx, hy = int(hand_point.x * frame_width), int(hand_point.y * frame_height)
                
                # Comprobar proximidad con cada ojo
                if np.linalg.norm([hx - left_eye_center[0], hy - left_eye_center[1]]) < detection_radius:
                    left_eye_rub = True
                if np.linalg.norm([hx - right_eye_center[0], hy - right_eye_center[1]]) < detection_radius:
                    right_eye_rub = True
        
        if left_eye_rub and right_eye_rub:
            return 3  # Ambos ojos
        elif left_eye_rub:
            return 1  # Ojo izquierdo
        elif right_eye_rub:
            return 2  # Ojo derecho
        else:
            return 0

    except Exception as e:
        logger.error(f"Error en detect_eye_rubbing: {e}")
        return 0

def detect_stretching(pose_landmarks: Any, config: Dict) -> bool:
    """
    Detecta si el usuario se está estirando, basándose en la postura de los brazos.
    Un estiramiento se considera si los brazos están extendidos por encima de los hombros.

    Args:
        pose_landmarks (Any): Objeto de landmarks de pose de MediaPipe.
        config (Dict): Diccionario de configuración con 'stretching_config'.

    Returns:
        bool: True si se detecta un estiramiento, False en caso contrario.
    """
    if not pose_landmarks:
        return False

    try:
        stretch_config = config.get('stretching_config', {})
        arm_stretch_threshold = stretch_config.get('ARM_STRETCH_THRESHOLD', 150)

        landmarks = pose_landmarks.landmark
        
        # Puntos clave para brazos y hombros
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

        # Condición 1: Al menos una muñeca debe estar por encima de su hombro
        left_wrist_above_shoulder = left_wrist.y < left_shoulder.y
        right_wrist_above_shoulder = right_wrist.y < right_shoulder.y

        if not (left_wrist_above_shoulder or right_wrist_above_shoulder):
            return False

        # Condición 2: Al menos un brazo debe estar relativamente estirado
        left_arm_angle = abs(calculate_angle(
            [left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y]
        ))
        right_arm_angle = abs(calculate_angle(
            [right_shoulder.x, right_shoulder.y], [right_elbow.x, right_elbow.y], [right_wrist.x, right_wrist.y]
        ))
        
        is_left_arm_stretched = left_arm_angle > arm_stretch_threshold
        is_right_arm_stretched = right_arm_angle > arm_stretch_threshold
        
        if is_left_arm_stretched or is_right_arm_stretched:
            logger.info("Patrón de estiramiento detectado: brazos elevados y extendidos.")
            return True
        
        return False

    except Exception as e:
        logger.error(f"Error en detect_stretching: {e}")
        return False