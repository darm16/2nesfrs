# logging_setup.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Configura el sistema de logging para escribir en la consola y en un archivo con rotación.
    Esta función es idempotente, lo que significa que se puede llamar de forma segura varias veces.
    """
    log_directory = "logs"
    try:
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
    except OSError as e:
        print(f"Error crítico: No se pudo crear el directorio de logs '{log_directory}'. {e}")
        # En este punto, la aplicación podría no continuar, pero el logging a consola aún funcionará.

    log_file_path = os.path.join(log_directory, "copiloto_id.log")

    # Obtener un logger con un nombre específico para nuestra aplicación
    logger = logging.getLogger("CopilotoIDApp")
    logger.setLevel(logging.INFO) # Nivel mínimo de mensajes a procesar

    # Evitar añadir handlers duplicados si la función se llama más de una vez
    if logger.hasHandlers():
        return logger

    # Formato estándar para los mensajes de log
    log_format = '%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)

    # 1. Handler para la consola (para depuración en vivo)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Handler para el archivo con rotación
    # Crea hasta 5 archivos de log de 5MB cada uno. Cuando el actual se llena,
    # se renombra a .1, el .1 a .2, y así sucesivamente.
    try:
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"No se pudo configurar el logging a archivo en '{log_file_path}': {e}")

    logger.info("Sistema de logging configurado.")
    return logger

# Instancia global del logger para ser importada por otros módulos
logger = setup_logging()