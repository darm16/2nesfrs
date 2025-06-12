# modules/hardware/mpu6050.py

import smbus2 as smbus
from logging_setup import logger

class MPU6050:
    """Clase para interactuar con el sensor MPU6050 a través de I2C."""
    def __init__(self, address=0x68):
        self.bus = None
        self.address = address
        try:
            self.bus = smbus.SMBus(1)  # 1 para Raspberry Pi
            self.bus.write_byte_data(self.address, 0x6B, 0) # Salir del modo de reposo
            logger.info("Sensor MPU6050 inicializado correctamente.")
        except Exception as e:
            logger.warning(f"No se pudo inicializar el sensor MPU6050 en la dirección {hex(address)}. El modo Sleep no funcionará. Error: {e}")
            self.bus = None

    def read_sensor_data(self):
        """Lee y devuelve los valores del acelerómetro y giroscopio."""
        if not self.bus: return None
        try:
            # Lectura de acelerómetro
            accel_x = self._read_word_2c(0x3B) / 16384.0
            accel_y = self._read_word_2c(0x3D) / 16384.0
            accel_z = self._read_word_2c(0x3F) / 16384.0
            # Lectura de giroscopio
            gyro_x = self._read_word_2c(0x43) / 131.0
            gyro_y = self._read_word_2c(0x45) / 131.0
            gyro_z = self._read_word_2c(0x47) / 131.0
            return (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
        except Exception as e:
            logger.error(f"Error al leer datos del MPU6050: {e}")
            return None

    def _read_word_2c(self, addr):
        """Lee un valor de 16 bits (complemento a dos) de una dirección."""
        high = self.bus.read_byte_data(self.address, addr)
        low = self.bus.read_byte_data(self.address, addr + 1)
        val = (high << 8) + low
        return val - 65536 if val >= 0x8000 else val

    def detect_motion(self, accel_threshold, gyro_threshold) -> bool:
        """Detecta si hay movimiento por encima de los umbrales."""
        data = self.read_sensor_data()
        if not data: return False
        
        accel_mag = (data[0]**2 + data[1]**2 + data[2]**2)**0.5
        gyro_mag = (data[3]**2 + data[4]**2 + data[5]**2)**0.5
        
        # Consideramos movimiento si el acelerómetro (sin gravedad) o el giroscopio se activan
        return (abs(accel_mag - 1) > accel_threshold or gyro_mag > gyro_threshold)