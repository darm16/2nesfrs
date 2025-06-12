# mpu_thread.py

import time
from PyQt5.QtCore import QThread, pyqtSignal
from logging_setup import logger
from mpu6050 import MPU6050

class MPUThread(QThread):
    """Hilo para monitorear el sensor MPU6050 en segundo plano."""
    motion_detected = pyqtSignal()
    no_motion_detected_for_duration = pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config.get('hardware_settings', {})
        self.sensor = MPU6050()
        self.running = False
        self.no_motion_timer_start = None

    def run(self):
        if not self.sensor.bus:
            logger.error("MPUThread no puede iniciar, el sensor no está disponible.")
            return
            
        self.running = True
        logger.info("MPUThread iniciado.")
        
        sleep_duration = self.config.get('mpu_sleep_threshold_minutes', 5) * 60
        accel_thresh = self.config.get('mpu_accel_threshold', 0.5)
        gyro_thresh = self.config.get('mpu_gyro_threshold', 0.5)

        while self.running:
            motion = self.sensor.detect_motion(accel_thresh, gyro_thresh)
            
            if motion:
                self.motion_detected.emit()
                self.no_motion_timer_start = None # Resetear el temporizador de inactividad
            else:
                if self.no_motion_timer_start is None:
                    self.no_motion_timer_start = time.time()
                
                # Comprobar si ha pasado el tiempo de inactividad
                if (time.time() - self.no_motion_timer_start) > sleep_duration:
                    self.no_motion_detected_for_duration.emit()
                    self.no_motion_timer_start = None # Resetear para no emitir la señal continuamente
            
            time.sleep(1) # El MPU no necesita ser sondeado a 30fps

    def stop(self):
        self.running = False