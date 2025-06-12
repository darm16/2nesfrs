# camera_thread.py

import cv2
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from logging_setup import logger
from config_manager import ConfigManager
import roi_autoexp

class CameraThread(QThread):
    """
    Hilo dedicado para la captura de frames de la cámara sin bloquear la interfaz gráfica.
    Emite cada fotograma capturado a través de una señal.
    """
    # Señal que emite un fotograma capturado como un array de numpy
    update_frame = pyqtSignal(np.ndarray)
    # Señal que emite un mensaje de error si la cámara falla
    error = pyqtSignal(str)

    def __init__(self, camera_index: int, parent=None, frame_width: int = None, frame_height: int = None):
        """
        Inicializa el hilo de la cámara.

        Args:
            camera_index (int): El índice de la cámara a utilizar.
            parent (QObject, optional): El objeto padre en la jerarquía de Qt.
            frame_width (int, optional): Ancho del frame deseado. Si no se especifica, se usa el de la configuración.
            frame_height (int, optional): Alto del frame deseado. Si no se especifica, se usa el de la configuración.
        """
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None

        # Cargar configuración de la cámara
        config = ConfigManager.load_full_config()
        
        # Usar la resolución proporcionada o la de la configuración
        if frame_width is not None and frame_height is not None:
            self.capture_resolution = (frame_width, frame_height)
        else:
            self.capture_resolution = (
                config['camera_settings'].get('FRAME_WIDTH', 1280),
                config['camera_settings'].get('FRAME_HEIGHT', 720)
            )
            
        self.target_fps = 20  # FPS objetivo para no sobrecargar la CPU
        self.frame_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0.05

    def run(self):
        """
        Método principal del hilo. Contiene el bucle de captura de frames.
        Este método se ejecuta cuando se llama a .start() en el hilo.
        """
        self.running = True
        logger.info(f"Iniciando hilo para cámara índice {self.camera_index} a {self.target_fps} FPS objetivo.")

        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap or not self.cap.isOpened():
                error_msg = f"No se pudo abrir la cámara con índice {self.camera_index}. " \
                            "Verifique si está conectada o en uso por otra aplicación."
                self.error.emit(error_msg)
                self.running = False
                logger.error(error_msg)
                return

            # Configurar propiedades de la cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_resolution[1])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) # Limitar el buffer para obtener frames más recientes

            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Cámara {self.camera_index} abierta. Resolución real: {int(actual_w)}x{int(actual_h)}")

            read_errors = 0
            frame_counter = 0
            while self.running:
                start_time = time.perf_counter()

                ret, frame = self.cap.read()

                if not ret:
                    read_errors += 1
                    logger.warning(f"Error al leer fotograma de cámara {self.camera_index} (Intento #{read_errors})")
                    if read_errors > self.target_fps * 5: # Si falla por 5 segundos seguidos
                        error_msg = f"Error persistente de lectura en cámara {self.camera_index}. La cámara podría estar desconectada."
                        self.error.emit(error_msg)
                        logger.error(error_msg)
                        break
                    time.sleep(0.1)
                    continue
                
                read_errors = 0
                frame_counter += 1

                # Verificar centrado cada 300 frames (~15 segundos a 20fps)
                if frame_counter % 300 == 0 and hasattr(roi_autoexp, 'auto_center_enabled'):
                    if roi_autoexp.auto_center_enabled:
                        height, width = frame.shape[:2]
                        roi_autoexp.ensure_roi_centered(width, height)

                # Emitir el fotograma volteado horizontalmente (efecto espejo para el usuario)
                self.update_frame.emit(cv2.flip(frame, 1))

                # Controlar los FPS para no consumir 100% de la CPU
                elapsed_time = time.perf_counter() - start_time
                sleep_time = self.frame_delay - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            error_msg = f"Excepción inesperada en el hilo de la cámara: {e}"
            self.error.emit(error_msg)
            logger.critical(error_msg, exc_info=True)
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            logger.info(f"Hilo para cámara {self.camera_index} detenido y recursos liberados.")
            self.running = False

    def stop(self):
        """
        Solicita la detención segura del bucle de captura del hilo.
        """
        logger.info(f"Solicitando detención del hilo de la cámara {self.camera_index}.")
        self.running = False

    def __del__(self):
        """
        Destructor para asegurar que la cámara se libere si el objeto se elimina.
        """
        if self.isRunning():
            self.stop()
            self.wait() # Esperar a que el hilo termine limpiamente