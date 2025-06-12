# fatigue_processor.py

import numpy as np
import time
import math
import os
from typing import Tuple, Optional, List, Dict, Any

# Importar el clasificador LSTM
try:
    from lstm_classifier import LSTMClassifier
    LSTM_AVAILABLE = True
except ImportError:
    logger.warning("No se pudo importar LSTMClassifier. La clasificación LSTM no estará disponible.")
    LSTM_AVAILABLE = False

# Se asume que existe un módulo de logging configurado.
# Si no, se usa el logging estándar como fallback.
try:
    from logging_setup import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Importar módulos locales
from nod_detector import NodDetector
from deteccion import (
    calculate_head_angle,
    detect_eye_rubbing,
    detect_stretching
)

class PausaActivaHandler:
    """
    Gestiona el tiempo para las pausas activas, con lógica de pausa y reseteo
    basada en la presencia robusta del usuario (rostro O cuerpo).
    """
    def __init__(self, work_duration_secs: int = 3600, break_reset_threshold_secs: int = 180):
        """
        Args:
            work_duration_secs (int): Ciclo de trabajo antes de una alerta (3600s = 1 hora).
            break_reset_threshold_secs (int): Tiempo de ausencia para resetear el contador (180s = 3 min).
        """
        self.work_duration = work_duration_secs
        self.reset_threshold = break_reset_threshold_secs
        
        self.elapsed_work_time: float = 0.0
        self.is_paused: bool = True
        self.pause_start_time: Optional[float] = None
        self.last_update_time: float = time.time()
        self.last_pausa_activa_time: Optional[float] = None
        logger.info(f"PausaActivaHandler inicializado. Umbral de reseteo: {self.reset_threshold}s.")

    def reset(self):
        """Resetea completamente el estado para un nuevo ciclo de trabajo."""
        self.elapsed_work_time = 0.0
        self.is_paused = True
        self.pause_start_time = None
        self.last_update_time = time.time()
        self.last_pausa_activa_time = time.time()
        logger.info("Manejador de Pausa Activa reseteado.")

    def update(self, face_detected: bool, pose_detected: bool) -> Tuple[bool, Optional[int]]:
        """Actualiza el estado del contador basado en la presencia del usuario."""
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        is_user_present = face_detected or pose_detected
        
        should_alert = False

        if is_user_present:
            if self.is_paused:
                logger.info("Presencia de usuario detectada. Contador de Pausa Activa reanudado.")
                self.is_paused = False
                self.pause_start_time = None

            self.elapsed_work_time += delta_time
            
            if self.elapsed_work_time >= self.work_duration:
                if self.last_pausa_activa_time is None or \
                   (current_time - self.last_pausa_activa_time) > self.work_duration * 0.95:
                    logger.warning("Alerta de Pausa Activa generada por tiempo de trabajo acumulado.")
                    should_alert = True
                    self.reset()
        
        else: # Si el usuario NO está presente (ni rostro, ni cuerpo)
            if not self.is_paused:
                self.is_paused = True
                self.pause_start_time = current_time
                logger.info("Usuario ausente. Contador de Pausa Activa pausado.")
            
            if self.pause_start_time and (current_time - self.pause_start_time) > self.reset_threshold:
                logger.info(f"Ausencia > {self.reset_threshold}s. Reseteando contador de Pausa Activa.")
                self.reset()

        time_remaining = self.work_duration - self.elapsed_work_time
        return should_alert, int(max(0, time_remaining))


class FatigueProcessor:
    """
    Encapsula toda la lógica para la detección de fatiga, distracción y otros
    comportamientos. Es el cerebro del monitoreo de comportamiento.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.nod_detector = NodDetector(config.get('nod_detector_config', {}))
        
        # Inicializar el clasificador LSTM si está habilitado
        self.lstm_classifier = None
        if config.get('use_lstm_classification', False) and LSTM_AVAILABLE:
            model_path = config.get('lstm_model_path', 'models/D.pth')
            if os.path.exists(model_path):
                try:
                    self.lstm_classifier = LSTMClassifier(model_path)
                    logger.info(f"Clasificador LSTM cargado desde {model_path}")
                except Exception as e:
                    logger.error(f"Error al cargar el clasificador LSTM: {e}")
            else:
                logger.warning(f"No se encontró el modelo LSTM en {model_path}. La clasificación LSTM no estará disponible.")
        
        # --- LECTURA DE CONFIGURACIÓN ANIDADA (VERSIÓN FINAL) ---
        pausa_config = config.get('pausa_activa_settings', {})
        self.pausa_handler = PausaActivaHandler(
            work_duration_secs=pausa_config.get('work_duration_seconds', 3600),
            break_reset_threshold_secs=pausa_config.get('reset_threshold_seconds', 180)
        )
        
        self.calibration_data: Optional[Dict] = None
        self.state: Dict[str, Any] = {}
        self.reset_state()

    def set_calibration(self, calibration_data: Dict):
        """Establece los datos de calibración para la sesión de inferencia."""
        self.calibration_data = calibration_data
        logger.info(f"FatigueProcessor calibrado con datos: {calibration_data}")
        self.reset_state()

    def reset_state(self):
        """Reinicia el estado interno para una nueva sesión de monitoreo."""
        self.state = {
            'yawn_frames': 0, 'closed_eyes_frames': 0, 'stretching_frames': 0,
            'eye_rubbing_frames': 0, 'distraction_frames': 0,
            'last_yawn_time': 0, 'last_stretching_time': 0, 'last_eye_rub_time': 0,
            'last_distraction_alert_time': 0, 'last_despierta_alert_time': 0,
        }
        self.nod_detector.reset()
        self.pausa_handler.reset()
        logger.info("Estado de FatigueProcessor reseteado.")

    def _calculate_facial_metrics(self, face_landmarks: Any, frame_shape: Tuple[int, int], calibration_data: Optional[Dict] = None) -> Dict:
        """Calcula todas las métricas faciales (EAR, MAR, PUC, MOE) a partir de los landmarks."""
        h, w = frame_shape
        lm_np = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
        
        def _distance(p1, p2): return np.linalg.norm(p1 - p2)
        
        def _eye_aspect_ratio(eye_points):
            A = _distance(eye_points[1], eye_points[5])
            B = _distance(eye_points[2], eye_points[4])
            C = _distance(eye_points[0], eye_points[3])
            return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

        left_eye_pts = lm_np[[362, 385, 387, 263, 373, 380]]
        right_eye_pts = lm_np[[33, 160, 158, 133, 153, 144]]
        ear = (_eye_aspect_ratio(left_eye_pts) + _eye_aspect_ratio(right_eye_pts)) / 2.0
        
        mouth_pts = lm_np[[61, 291, 0, 17]]
        mar = _distance(mouth_pts[0], mouth_pts[1]) / _distance(mouth_pts[2], mouth_pts[3])

        def _calculate_circularity(iris_landmarks_np):
            horizontal_radius = _distance(iris_landmarks_np[2], iris_landmarks_np[4]) / 2.0
            vertical_radius = _distance(iris_landmarks_np[1], iris_landmarks_np[3]) / 2.0
            if horizontal_radius < 1e-6 or vertical_radius < 1e-6: return 0.0
            area = math.pi * horizontal_radius * vertical_radius
            a, b = horizontal_radius, vertical_radius
            perimeter = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
            if perimeter < 1e-6: return 0.0
            circularity = (4 * math.pi * area) / (perimeter ** 2)
            return np.clip(circularity, 0.0, 1.0)

        puc = 0.0
        try:
            right_iris_pts = lm_np[[473, 474, 475, 476, 477]]
            left_iris_pts = lm_np[[468, 469, 470, 471, 472]]
            puc_right = _calculate_circularity(right_iris_pts)
            puc_left = _calculate_circularity(left_iris_pts)
            puc = (puc_right + puc_left) / 2.0 if puc_right > 0 and puc_left > 0 else max(puc_right, puc_left)
        except IndexError:
            logger.warning("Landmarks de iris no disponibles para cálculo de PUC.")
            puc = 0.0
            
        moe = 0.0
        if calibration_data and 'cal_ear_mean' in calibration_data and 'cal_mar_mean' in calibration_data:
            cal_ear, cal_mar = calibration_data['cal_ear_mean'], calibration_data['cal_mar_mean']
            ear_norm = ear / cal_ear if cal_ear > 1e-6 else 1.0
            mar_norm = mar / cal_mar if cal_mar > 1e-6 else 1.0
            moe = mar_norm / ear_norm if ear_norm > 1e-6 else mar_norm
        else:
            moe = mar / ear if ear > 1e-6 else mar

        return {'ear': ear, 'mar': mar, 'puc': puc, 'moe': moe, 'landmarks_np': lm_np}

    def process_frame_for_calibration(self, face_results: Any, frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """Procesa un frame para extraer métricas que serán usadas en la calibración."""
        if not face_results or not face_results.multi_face_landmarks:
            return None
        metrics = self._calculate_facial_metrics(face_results.multi_face_landmarks[0], frame_shape)
        return {k: v for k, v in metrics.items() if k != 'landmarks_np'}

    def process_frame_for_inference(self, frame: np.ndarray, face_results: Any, hand_results: Any, pose_results: Any) -> Tuple[List[str], Dict]:
        """Método principal. Procesa un frame para detectar eventos de fatiga."""
        detected_events = []
        overlay_data = {'metrics': {}, 'alert_text': None, 'is_max_alert': False, 'pausa_text': None}
        frame_h, frame_w, _ = frame.shape
        
        face_detected = bool(face_results and face_results.multi_face_landmarks)
        pose_detected = bool(pose_results and pose_results.pose_landmarks)

        alert_pausa, time_remaining = self.pausa_handler.update(face_detected, pose_detected)
        if alert_pausa: detected_events.append("Pausa Activa")
        if time_remaining is not None:
            mins, secs = divmod(time_remaining, 60)
            overlay_data['pausa_text'] = f"Pausa en: {mins:02d}:{secs:02d}"

        if not face_detected:
            self.state['distraction_frames'] += 1
            threshold = self.config.get('fatigue_detection_thresholds', {}).get('DISTRACTION_FRAMES_THRESHOLD', 30)
            if self.state['distraction_frames'] > threshold:
                if time.time() - self.state['last_distraction_alert_time'] > 5:
                    detected_events.append("Distraccion")
                    self.state['last_distraction_alert_time'] = time.time()
            overlay_data['alert_text'] = "ROSTRO NO DETECTADO"
            return list(set(detected_events)), overlay_data
        
        self.state['distraction_frames'] = 0
        face_landmarks_obj = face_results.multi_face_landmarks[0]
        
        metrics = self._calculate_facial_metrics(face_landmarks_obj, (frame_h, frame_w), self.calibration_data)
        overlay_data['metrics'] = {k: v for k, v in metrics.items() if k != 'landmarks_np'}
        
        # Usar el clasificador LSTM si está disponible
        if self.lstm_classifier is not None:
            lstm_prediction = self.lstm_classifier.update(
                metrics['ear'], 
                metrics['mar'], 
                metrics['puc'], 
                metrics['moe']
            )
            if lstm_prediction is not None:
                overlay_data['lstm_prediction'] = lstm_prediction
                if lstm_prediction == 1 and current_time - self.state.get('last_lstm_alert_time', 0) > 30:
                    detected_events.append("Somnolencia")
                    self.state['last_lstm_alert_time'] = current_time
                    logger.info("Detección de somnolencia por LSTM")
        
        if self.calibration_data is None:
            return ["AWAITING_CALIBRATION"], overlay_data
        
        angle_x, angle_y = calculate_head_angle(face_landmarks_obj)
        overlay_data['angles'] = {'x': angle_x or 0.0, 'y': angle_y or 0.0}

        thresholds = self.config.get('fatigue_detection_thresholds', {})
        cal_data = self.calibration_data
        current_time = time.time()

        if metrics['mar'] > cal_data['cal_mar_mean'] * thresholds.get('YAWN_MAR_FACTOR', 1.8):
            self.state['yawn_frames'] += 1
            if self.state['yawn_frames'] > thresholds.get('YAWN_FRAMES_THRESHOLD', 8):
                if current_time - self.state['last_yawn_time'] > 10:
                    detected_events.append("Bostezar")
                    self.state['last_yawn_time'] = current_time
                    self.state['yawn_frames'] = 0
        else:
            self.state['yawn_frames'] = 0
            
        ear_threshold = cal_data['cal_ear_mean'] * thresholds.get('DROWSINESS_EAR_FACTOR', 0.75)
        if metrics['ear'] < ear_threshold:
            self.state['closed_eyes_frames'] += 1
            if self.state['closed_eyes_frames'] > thresholds.get('MAX_ALERT_EYES_CLOSED_FRAMES', 60):
                if current_time - self.state['last_despierta_alert_time'] > 5:
                    detected_events.append("Despierta")
                    self.state['last_despierta_alert_time'] = current_time
            elif self.state['closed_eyes_frames'] > thresholds.get('DROWSINESS_EYES_CLOSED_FRAMES', 20):
                detected_events.append("Somnolencia")
        else:
            self.state['closed_eyes_frames'] = 0

        is_nodding, _ = self.nod_detector.update(angle_y or 0.0)
        if is_nodding:
            detected_events.append("Cabeceo")
            
        rub_type = detect_eye_rubbing(metrics.get('landmarks_np'), hand_results, self.config)
        if rub_type > 0:
            self.state['eye_rubbing_frames'] += 1
            if self.state['eye_rubbing_frames'] > thresholds.get('EYE_RUBBING_FRAMES', 15):
                if current_time - self.state['last_eye_rub_time'] > 10:
                    detected_events.append("Frotar Ojos")
                    self.state['last_eye_rub_time'] = current_time
        else:
            self.state['eye_rubbing_frames'] = 0

        if "Despierta" in detected_events or "Cabeceo" in detected_events:
            overlay_data['alert_text'] = "¡DESPIERTA!" if "Despierta" in detected_events else "NO TE DUERMAS"
            overlay_data['is_max_alert'] = True
        elif "Somnolencia" in detected_events:
            overlay_data['alert_text'] = "SOMNOLENCIA DETECTADA"
        elif "Bostezar" in detected_events:
            overlay_data['alert_text'] = "BOSTEZO DETECTADO"
        elif "Frotar Ojos" in detected_events:
            overlay_data['alert_text'] = "DESCANSA LA VISTA"
        elif "Distraccion" in detected_events:
            overlay_data['alert_text'] = "ATENCION AL FRENTE"
        elif "Pausa Activa" in detected_events:
            overlay_data['alert_text'] = "PAUSA ACTIVA RECOMENDADA"
            
        return list(set(detected_events)), overlay_data