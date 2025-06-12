# config_manager.py

import json
import os
import hashlib
import time
from logging_setup import logger
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Gestiona la carga y guardado del archivo de configuración JSON.
    Centraliza todos los parámetros de la aplicación y la lógica de seguridad de claves.
    """
    
    CONFIG_FILE = "config.json"

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Devuelve la estructura de configuración completa con valores por defecto."""
        return {
            "camera_settings": {
                "camera_index": 0,
                "FRAME_WIDTH": 1280,
                "FRAME_HEIGHT": 720,
                "MAX_CAMERA_INDEX_TO_CHECK": 4
            },
            "recognition_settings": {
                "model_filename": "w600k_mbf.onnx",
                "model_name": "ArcFace Buffalo S",
                "threshold": 0.45,
                "mode": "multi-pose",
                "pose_limits": {
                    "MAX_ABS_PITCH": 20.0,
                    "MAX_ABS_YAW": 25.0,
                    "MAX_ABS_ROLL": 20.0
                }
            },
            "fatigue_detection_thresholds": {
                "EAR_THRESHOLD": 0.23,
                "MOE_THRESHOLD": 0.4,
                "YAWN_FRAMES_THRESHOLD": 22,
                "DISTRACTION_FRAMES_THRESHOLD": 60,
                "STATIC_POSITION_SECONDS": 3600,
                "EYE_RUBBING_FRAMES": 15
            },
            "event_toggles": {
                "Despierta": True, "Distraccion": True, "Somnolencia": True,
                "Estiramiento": True, "Posicion Estatica": True, "Frotar Ojos": True,
                "Cabeceo": True, "Bostezar": True
            },
            "voice_alerts": {
                "Despierta": {"enabled": True, "text": "Despierta. Mantente alerta.", "priority": 0},
                "Cabeceo": {"enabled": True, "text": "Peligro de microsueño detectado.", "priority": 0},
                "Distraccion": {"enabled": True, "text": "Atención a la carretera.", "priority": 1},
                "Somnolencia": {"enabled": True, "text": "Se detecta somnolencia.", "priority": 1},
                "Bostezar": {"enabled": True, "text": "Parece que estás cansado.", "priority": 2},
                "Frotar Ojos": {"enabled": True, "text": "Descansa la vista.", "priority": 2},
                "Posicion Estatica": {"enabled": True, "text": "Llevas mucho tiempo en la misma posición, considera moverte.", "priority": 3},
                "Estiramiento": {"enabled": True, "text": "Un estiramiento es una buena idea.", "priority": 3}
            },
            "privacy_settings": {
                "privacy_mode_default": True,
                "photo_view_password_hash": "",
                "photo_view_salt": "",
                "recovery_code_hash": ""
            },
            "state_machine_thresholds": {
                "stable_face_frames_to_identify": 90,
                "face_lost_frames_to_logout": 900,
                "auto_register_seconds": 15
            },
            "profile_enrichment": {
                "enabled": True,
                "embeddings_per_session_target": 5,
                "min_pose_difference_threshold": 15.0
            },
            "hardware_settings": {
                "USE_MPU": False,
                "mpu_sleep_threshold_minutes": 5,
                "mpu_accel_threshold": 0.5,
                "mpu_gyro_threshold": 0.5
            },
            "pausa_activa_settings": {
                "work_duration_seconds": 3600,
                "reset_threshold_seconds": 180
            },
            "reporting_settings": {
                "SAVE_LOG_ON_SESSION_END": True
            },
            "roi_autoexposure_settings": {
                "ROI_ENABLED": True,
                "EXPOSURE_TARGET": 127,
                "GAMMA_VALUE": 1.1,
                "AUTO_CENTER_DEFAULT": True,
                "CENTER_CONFIG": {
                    "ROI_WIDTH_PERCENT": 0.6,
                    "ROI_HEIGHT_PERCENT": 0.6,
                    "MIN_ROI_WIDTH": 200,
                    "MIN_ROI_HEIGHT": 150,
                    "MAX_ROI_WIDTH": 800,
                    "MAX_ROI_HEIGHT": 600
                }
            },
            "lstm_model_path": "models/D.pth",
            "use_lstm_classification": True,
            "zoom_settings": {
                "ZOOM_FACTOR_CALIBRATION": 1.5,
                "ZOOM_FACTOR_INFERENCE": 1.5,
                "ZOOM_POSITION": "top-right",
                "RESIZE_ZOOM_BOX": True
            },
            "drowsiness_levels": {
                "LEVE": 60,
                "MODERADO": 90,
                "SEVERO": 120
            }
        }

    @staticmethod
    def setup_config_file() -> bool:
        """
        Asegura que el config.json exista y esté completo. Si no existe o está corrupto,
        lo crea con valores por defecto. Si existe pero le faltan claves, las añade.
        Devuelve True si se necesita configurar una clave inicial.
        """
        default_config = ConfigManager.get_default_config()
        if not os.path.exists(ConfigManager.CONFIG_FILE):
            logger.info("No se encontró config.json. Creando uno nuevo con valores por defecto.")
            ConfigManager.save_full_config(default_config)
            return True

        try:
            with open(ConfigManager.CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.error(f"'{ConfigManager.CONFIG_FILE}' está corrupto. Se creará uno nuevo y se respaldará el antiguo.")
            os.rename(ConfigManager.CONFIG_FILE, f"{ConfigManager.CONFIG_FILE}.{int(time.time())}.bak")
            ConfigManager.save_full_config(default_config)
            return True
        
        needs_update = False
        def update_dict_recursively(default, user):
            nonlocal needs_update
            for key, value in default.items():
                if key not in user:
                    user[key] = value
                    needs_update = True
                elif isinstance(value, dict) and isinstance(user.get(key), dict):
                    update_dict_recursively(value, user[key])
        
        update_dict_recursively(default_config, user_config)

        if needs_update:
            logger.info("Actualizando config.json con nuevas claves y secciones por defecto.")
            ConfigManager.save_full_config(user_config)
        
        return not user_config.get('privacy_settings', {}).get('photo_view_password_hash')

    @staticmethod
    def load_full_config() -> Dict[str, Any]:
        """Carga el archivo de configuración completo desde config.json."""
        try:
            with open(ConfigManager.CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"'{ConfigManager.CONFIG_FILE}' no encontrado o corrupto: {e}. Usando configuración por defecto.")
            return ConfigManager.get_default_config()

    @staticmethod
    def save_full_config(config_data: Dict[str, Any]):
        """Guarda un diccionario de configuración completo en el archivo config.json."""
        try:
            with open(ConfigManager.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Configuración guardada exitosamente en '{ConfigManager.CONFIG_FILE}'.")
        except Exception as e:
            logger.error(f"Error crítico al guardar en '{ConfigManager.CONFIG_FILE}': {e}")

    @staticmethod
    def set_new_password(password: str) -> str:
        """Genera hash y sal para una nueva clave, la guarda y devuelve un código de recuperación."""
        config = ConfigManager.load_full_config()
        salt = os.urandom(16).hex()
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
        
        recovery_code = f"CID-{os.urandom(2).hex().upper()}-{os.urandom(2).hex().upper()}"
        hashed_recovery_code = hashlib.sha256(recovery_code.encode('utf-8')).hexdigest()

        if 'privacy_settings' not in config:
            config['privacy_settings'] = {}
            
        config['privacy_settings']['photo_view_password_hash'] = hashed_password
        config['privacy_settings']['photo_view_salt'] = salt
        config['privacy_settings']['recovery_code_hash'] = hashed_recovery_code
        
        ConfigManager.save_full_config(config)
        logger.info("Nueva clave de privacidad y código de recuperación han sido establecidos y guardados.")
        return recovery_code

    @staticmethod
    def verify_password(password_to_check: str) -> bool:
        """Verifica una clave introducida contra el hash almacenado."""
        config = ConfigManager.load_full_config()
        privacy_settings = config.get('privacy_settings', {})
        stored_hash = privacy_settings.get('photo_view_password_hash', '')
        stored_salt = privacy_settings.get('photo_view_salt', '')
        
        if not stored_hash or not stored_salt:
            logger.warning("Intento de verificar clave, pero no hay ninguna configurada.")
            return False
        
        try:
            current_hash = hashlib.pbkdf2_hmac('sha256', password_to_check.encode('utf-8'), stored_salt.encode('utf-8'), 100000).hex()
            return current_hash == stored_hash
        except Exception as e:
            logger.error(f"Error durante la verificación de clave: {e}")
            return False

    @staticmethod
    def verify_recovery_code(code_to_check: str) -> bool:
        """Verifica un código de recuperación contra su hash almacenado."""
        config = ConfigManager.load_full_config()
        stored_hash = config.get('privacy_settings', {}).get('recovery_code_hash', '')
        if not stored_hash: 
            logger.warning("Intento de usar código de recuperación, pero no hay ninguno configurado.")
            return False

        current_hash = hashlib.sha256(code_to_check.encode('utf-8')).hexdigest()
        return current_hash == stored_hash