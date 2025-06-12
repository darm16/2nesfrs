# modules/behavioral/roi_autoexp.py

import cv2
import numpy as np
import os
from logging_setup import logger

# --- Variables de Estado a Nivel de Módulo ---
# Aunque no es ideal, se mantiene esta estructura para preservar la lógica original.
# Una futura refactorización podría encapsular esto en una clase.

roi_selected = False
roi = None  # Formato: (x, y, w, h)
dragging = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
enable_roi_selection = False
DISPLAY_ONLY_ROI = False
auto_center_enabled = True  # Habilitar centrado automático por defecto

# Variables para el ajuste de exposición
error_integral = 0.0
last_error = 0.0
last_adjustment = 1.0

# Configuración por defecto del ROI centrado
DEFAULT_ROI_CONFIG = {
    "ROI_WIDTH_PERCENT": 0.6,
    "ROI_HEIGHT_PERCENT": 0.6,
    "MIN_ROI_WIDTH": 200,
    "MIN_ROI_HEIGHT": 150,
    "MAX_ROI_WIDTH": 800,
    "MAX_ROI_HEIGHT": 600
}

# Variables para centrado automático
auto_center_enabled = True
roi_center_config = {
    'ROI_WIDTH_PERCENT': 0.6,
    'ROI_HEIGHT_PERCENT': 0.6,
    'MIN_ROI_WIDTH': 200,
    'MIN_ROI_HEIGHT': 150,
    'MAX_ROI_WIDTH': 800,
    'MAX_ROI_HEIGHT': 600,
    'LOCK_ASPECT_RATIO': True,
    'PREFERRED_ASPECT_RATIO': 4/3
}

def init_roi_system():
    """Inicializa/resetea todas las variables de estado del módulo ROI."""
    global roi_selected, roi, dragging, top_left_pt, bottom_right_pt, enable_roi_selection, DISPLAY_ONLY_ROI, auto_center_enabled
    roi_selected = False
    roi = None
    dragging = False
    top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
    enable_roi_selection = False
    DISPLAY_ONLY_ROI = False
    auto_center_enabled = True
    logger.info("Sistema ROI inicializado a sus valores por defecto.")

def calculate_centered_roi_fixed(image_width: int, image_height: int) -> tuple:
    """
    Calcula un ROI perfectamente centrado en la imagen.
    
    Args:
        image_width: Ancho de la imagen
        image_height: Alto de la imagen
        
    Returns:
        tuple: (x, y, width, height) del ROI centrado
    """
    try:
        # Calcular dimensiones del ROI
        roi_width = int(image_width * roi_center_config['ROI_WIDTH_PERCENT'])
        roi_height = int(image_height * roi_center_config['ROI_HEIGHT_PERCENT'])
        
        # Aplicar límites mínimos y máximos
        roi_width = max(roi_center_config['MIN_ROI_WIDTH'], 
                       min(roi_width, roi_center_config['MAX_ROI_WIDTH']))
        roi_height = max(roi_center_config['MIN_ROI_HEIGHT'], 
                        min(roi_height, roi_center_config['MAX_ROI_HEIGHT']))
        
        # Mantener relación de aspecto si está habilitado
        if roi_center_config['LOCK_ASPECT_RATIO']:
            target_ratio = roi_center_config['PREFERRED_ASPECT_RATIO']
            current_ratio = roi_width / roi_height
            
            if current_ratio > target_ratio:
                roi_width = int(roi_height * target_ratio)
            else:
                roi_height = int(roi_width / target_ratio)
        
        # Asegurar que no exceda los límites de la imagen
        roi_width = min(roi_width, image_width - 10)
        roi_height = min(roi_height, image_height - 10)
        
        # Calcular posición centrada
        roi_x = (image_width - roi_width) // 2
        roi_y = (image_height - roi_height) // 2
        
        return (roi_x, roi_y, roi_width, roi_height)
        
    except Exception as e:
        logger.error(f"Error en calculate_centered_roi_fixed: {e}")
        # ROI de emergencia
        roi_width = int(image_width * 0.5)
        roi_height = int(image_height * 0.5)
        roi_x = (image_width - roi_width) // 2
        roi_y = (image_height - roi_height) // 2
        return (roi_x, roi_y, roi_width, roi_height)

def init_centered_roi(image_width: int, image_height: int) -> bool:
    """
    Inicializa un ROI centrado automáticamente.
    
    Args:
        image_width: Ancho de la imagen
        image_height: Alto de la imagen
        
    Returns:
        bool: True si se inicializó correctamente
    """
    global roi, roi_selected
    
    try:
        logger.info(f"Inicializando ROI centrado para resolución: {image_width}x{image_height}")
        
        # Calcular ROI centrado
        centered_roi = calculate_centered_roi_fixed(image_width, image_height)
        
        # Actualizar variables globales
        roi = centered_roi
        roi_selected = True
        
        logger.info(f"ROI centrado inicializado: {centered_roi}")
        return True
        
    except Exception as e:
        logger.error(f"Error al inicializar ROI centrado: {e}")
        return False

def ensure_roi_centered(image_width: int, image_height: int):
    """
    Asegura que el ROI esté centrado, creándolo si no existe.
    
    Args:
        image_width: Ancho de la imagen
        image_height: Alto de la imagen
    """
    global roi, roi_selected, auto_center_enabled
    
    if not auto_center_enabled:
        return
    
    try:
        # Si no hay ROI, crear uno centrado
        if not roi_selected or roi is None:
            init_centered_roi(image_width, image_height)
            return
        
        # Verificar si el ROI actual está centrado (con tolerancia)
        current_roi = roi
        centered_roi = calculate_centered_roi_fixed(image_width, image_height)
        
        # Tolerancia de 10 píxeles
        tolerance = 10
        if (abs(current_roi[0] - centered_roi[0]) > tolerance or
            abs(current_roi[1] - centered_roi[1]) > tolerance or
            abs(current_roi[2] - centered_roi[2]) > tolerance * 2 or
            abs(current_roi[3] - centered_roi[3]) > tolerance * 2):
            
            logger.info("ROI descentrado detectado, recentrando...")
            roi = centered_roi
            roi_selected = True
            
    except Exception as e:
        logger.error(f"Error al verificar centrado de ROI: {e}")

def get_roi_status() -> dict:
    """Devuelve el estado actual del ROI en un diccionario."""
    return {
        "roi": roi,
        "roi_selected": roi_selected,
        "display_only_roi": DISPLAY_ONLY_ROI,
        "enable_roi_selection": enable_roi_selection,
        "auto_center_enabled": auto_center_enabled
    }

def reset_roi():
    """Resetea completamente la selección de ROI."""
    init_roi_system()
    logger.info("Selección de ROI reseteada.")

def init_centered_roi(frame_width: int, frame_height: int, config: dict = None):
    """
    Inicializa un ROI centrado en el frame con las dimensiones especificadas.
    
    Args:
        frame_width: Ancho del frame de la cámara
        frame_height: Alto del frame de la cámara
        config: Diccionario con configuración del ROI (opcional)
    """
    global roi, roi_selected
    
    if config is None:
        config = DEFAULT_ROI_CONFIG
        
    # Calcular dimensiones del ROI basado en porcentajes
    roi_w = int(frame_width * config.get("ROI_WIDTH_PERCENT", 0.6))
    roi_h = int(frame_height * config.get("ROI_HEIGHT_PERCENT", 0.6))
    
    # Aplicar límites mínimos y máximos
    roi_w = max(config.get("MIN_ROI_WIDTH", 200), min(roi_w, config.get("MAX_ROI_WIDTH", 800)))
    roi_h = max(config.get("MIN_ROI_HEIGHT", 150), min(roi_h, config.get("MAX_ROI_HEIGHT", 600)))
    
    # Calcular posición centrada
    x = (frame_width - roi_w) // 2
    y = (frame_height - roi_h) // 2
    
    # Actualizar ROI
    roi = (x, y, roi_w, roi_h)
    roi_selected = True
    logger.info(f"ROI centrado inicializado en: {roi}")

def ensure_roi_centered(frame_width: int, frame_height: int, config: dict = None):
    """
    Asegura que el ROI esté centrado en el frame.
    
    Args:
        frame_width: Ancho del frame de la cámara
        frame_height: Alto del frame de la cámara
        config: Diccionario con configuración del ROI (opcional)
    """
    global roi, roi_selected, auto_center_enabled
    
    if not auto_center_enabled:
        return
        
    if not roi_selected or roi is None:
        init_centered_roi(frame_width, frame_height, config)
        return
        
    # Verificar si el ROI actual está centrado correctamente
    if config is None:
        config = DEFAULT_ROI_CONFIG
        
    # Calcular posición centrada esperada
    roi_w = int(frame_width * config.get("ROI_WIDTH_PERCENT", 0.6))
    roi_h = int(frame_height * config.get("ROI_HEIGHT_PERCENT", 0.6))
    roi_w = max(config.get("MIN_ROI_WIDTH", 200), min(roi_w, config.get("MAX_ROI_WIDTH", 800)))
    roi_h = max(config.get("MIN_ROI_HEIGHT", 150), min(roi_h, config.get("MAX_ROI_HEIGHT", 600)))
    
    expected_x = (frame_width - roi_w) // 2
    expected_y = (frame_height - roi_h) // 2
    
    # Si el ROI actual no está centrado, corregirlo
    if roi[0] != expected_x or roi[1] != expected_y or roi[2] != roi_w or roi[3] != roi_h:
        roi = (expected_x, expected_y, roi_w, roi_h)
        logger.debug(f"ROI re-centrado a: {roi}")

def toggle_auto_center(enabled: bool = None):
    """
    Habilita o deshabilita el centrado automático del ROI.
    
    Args:
        enabled: Si es None, alterna el estado actual. Si es bool, establece el estado.
    """
    global auto_center_enabled
    
    if enabled is None:
        auto_center_enabled = not auto_center_enabled
    else:
        auto_center_enabled = bool(enabled)
        
    logger.info(f"Centrado automático del ROI: {'Activado' if auto_center_enabled else 'Desactivado'}")
    return auto_center_enabled

def toggle_auto_center():
    """Activa/desactiva el centrado automático del ROI."""
    global auto_center_enabled
    auto_center_enabled = not auto_center_enabled
    logger.info(f"Centrado automático de ROI: {'activado' if auto_center_enabled else 'desactivado'}")
    return auto_center_enabled

def load_roi_from_file(roi_file_path: str):
    """Carga una selección de ROI desde un archivo."""
    global roi, roi_selected
    try:
        if os.path.exists(roi_file_path):
            with open(roi_file_path, "r") as f:
                parts = f.read().strip().split(',')
                if len(parts) == 4:
                    roi = tuple(map(int, parts))
                    roi_selected = True
                    logger.info(f"ROI cargado desde archivo {roi_file_path}: {roi}")
                    return True
    except Exception as e:
        logger.error(f"Error al cargar ROI desde archivo: {e}")
    return False

def save_roi_to_file(roi_file_path: str):
    """Guarda la selección de ROI actual a un archivo."""
    if roi:
        try:
            with open(roi_file_path, "w") as f:
                f.write(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}")
            logger.info(f"ROI guardado en archivo: {roi_file_path}")
        except Exception as e:
            logger.error(f"Error al guardar ROI en archivo: {e}")

def select_roi(event, x, y, flags, frame_shape):
    """
    Lógica para manejar los eventos del mouse y definir el ROI.
    Esta función está diseñada para ser llamada desde el manejador de eventos de la GUI.
    """
    global top_left_pt, bottom_right_pt, dragging, roi_selected, roi, enable_roi_selection, DISPLAY_ONLY_ROI, auto_center_enabled

    # Lógica para los botones virtuales dibujados en el frame
    # Botón ROI SELECT
    if 10 <= x <= 210 and 10 <= y <= 40 and event == cv2.EVENT_LBUTTONDOWN:
        enable_roi_selection = not enable_roi_selection
        logger.info(f"Selección de ROI cambiado a: {enable_roi_selection}")
        return
    
    # Botón ROI ONLY
    if 10 <= x <= 210 and 50 <= y <= 80 and event == cv2.EVENT_LBUTTONDOWN:
        if roi_selected:
            DISPLAY_ONLY_ROI = not DISPLAY_ONLY_ROI
            logger.info(f"Modo 'Display Only ROI' cambiado a: {DISPLAY_ONLY_ROI}")
        return
    
    # Botón CENTER ROI (nuevo)
    if 10 <= x <= 210 and 90 <= y <= 120 and event == cv2.EVENT_LBUTTONDOWN:
        if frame_shape and len(frame_shape) >= 2:
            height, width = frame_shape[:2]
            init_centered_roi(width, height)
            enable_roi_selection = False
            logger.info("ROI centrado mediante botón")
        return
    
    # Botón AUTO CENTER (nuevo)
    if 10 <= x <= 210 and 130 <= y <= 160 and event == cv2.EVENT_LBUTTONDOWN:
        auto_center_enabled = toggle_auto_center()
        return

    # Lógica de dibujo del rectángulo manual
    if enable_roi_selection:
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            top_left_pt = (x, y)
            bottom_right_pt = (x, y)
            auto_center_enabled = False  # Desactivar auto-centrado al seleccionar manualmente
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                bottom_right_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging:
                dragging = False
                roi_selected = True
                x1, y1 = min(top_left_pt[0], bottom_right_pt[0]), min(top_left_pt[1], bottom_right_pt[1])
                x2, y2 = max(top_left_pt[0], bottom_right_pt[0]), max(top_left_pt[1], bottom_right_pt[1])
                
                # Asegurar que el ROI tenga un tamaño mínimo
                if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
                    logger.warning("El ROI seleccionado es demasiado pequeño. Inténtelo de nuevo.")
                    roi_selected = False
                    return

                roi = (x1, y1, x2 - x1, y2 - y1)
                logger.info(f"Nuevo ROI seleccionado manualmente: {roi}")
                enable_roi_selection = False

def adjust_gamma(image, gamma=1.0):
    """Aplica corrección gamma a una imagen."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def smartphone_exposure_balance(frame, gamma_val):
    """Aplica un balance de exposición similar al de los smartphones modernos."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab = cv2.merge([l_clahe, a, b])
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return adjust_gamma(frame_enhanced, gamma=gamma_val)
    except cv2.error as e:
        logger.warning(f"Error en el procesamiento de color para balance de exposición: {e}")
        return frame

def adjust_exposure(frame, cap, config):
    """Ajusta la exposición de la cámara basándose en la luminosidad del ROI."""
    global error_integral, last_error, last_adjustment

    if not config.get('roi_autoexposure_settings', {}).get('ROI_ENABLED', False) or not roi_selected or roi is None:
        return frame

    try:
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        if roi_frame.size == 0:
            return frame

        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        luminosity = np.mean(gray_roi)

        target = config['roi_autoexposure_settings'].get('EXPOSURE_TARGET', 127)
        error = target - luminosity
        
        # Controlador P simple
        adjustment_factor = 1.0 + (error / 255.0) * 0.1

        # Suavizado del ajuste
        last_adjustment = last_adjustment * 0.9 + adjustment_factor * 0.1
        
        current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        new_exposure = current_exposure * last_adjustment
        
        # Limitar la exposición
        min_exp = -10
        max_exp = -4
        new_exposure = np.clip(new_exposure, min_exp, max_exp)
        
        cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)

        # Aplicar mejora de contraste al ROI
        gamma_val = config['roi_autoexposure_settings'].get('GAMMA_VALUE', 1.1)
        frame[y:y+h, x:x+w] = smartphone_exposure_balance(roi_frame, gamma_val)
        return frame

    except Exception as e:
        logger.error(f"Error durante el ajuste de exposición: {e}")
        return frame

def draw_roi_interface(frame):
    """Dibuja todos los componentes de la interfaz de ROI sobre el fotograma."""
    
    # Modo de vista enfocada en ROI
    if DISPLAY_ONLY_ROI and roi_selected and roi:
        x, y, w, h = roi
        # Asegurarse de que el recorte sea válido
        if w > 0 and h > 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
            cropped_frame = frame[y:y+h, x:x+w].copy()
            # Dibujar botón para salir de este modo
            cv2.rectangle(cropped_frame, (w-85, 5), (w-5, 35), (0,0,0), -1)
            cv2.rectangle(cropped_frame, (w-85, 5), (w-5, 35), (255,255,255), 1)
            cv2.putText(cropped_frame, "Vista Completa", (w-80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            return cropped_frame
        else:
            # Si el ROI es inválido, desactivar el modo
            reset_roi()

    # Modo de vista completa
    # Botón ROI SELECT
    color_select = (0, 255, 0) if enable_roi_selection else (100, 100, 100)
    cv2.rectangle(frame, (10, 10), (210, 40), color_select, -1)
    cv2.putText(frame, "1. Seleccionar ROI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Botón ROI ONLY
    color_focus = (0, 255, 255) if roi_selected else (100, 100, 100)
    cv2.rectangle(frame, (10, 50), (210, 80), color_focus, -1)
    cv2.putText(frame, "2. Vista Enfocada", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Botón CENTER ROI
    cv2.rectangle(frame, (10, 90), (210, 120), (255, 165, 0), -1)  # Naranja
    cv2.putText(frame, "3. Centrar ROI", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Botón AUTO CENTER
    color_auto = (0, 255, 0) if auto_center_enabled else (128, 128, 128)
    cv2.rectangle(frame, (10, 130), (210, 160), color_auto, -1)
    cv2.putText(frame, f"4. Auto-Centro: {'ON' if auto_center_enabled else 'OFF'}", (15, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Dibujar el rectángulo del ROI si está seleccionado
    if roi_selected and roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Dibujar líneas de centro
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.line(frame, (x, center_y), (x + w, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, y), (center_x, y + h), (0, 255, 255), 1)
        
        # Mostrar información del ROI
        status_text = "ROI Centrado" if auto_center_enabled else "ROI Manual"
        cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{w}x{h}", (x + w - 60, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Dibujar el rectángulo de selección mientras se arrastra
    if dragging:
        cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 255), 2)

    return frame