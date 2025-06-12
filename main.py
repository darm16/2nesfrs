# main.py

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

# --- Importar los módulos principales de gestión y la GUI ---
# Se importan aquí para que cualquier error de importación se detecte al inicio.
try:
    from logging_setup import logger
    from config_manager import ConfigManager
    from database_manager import DatabaseManager
    from gui import MainWindow
except ImportError as e:
    # Este es un error fatal si faltan archivos del proyecto.
    # Usamos un print porque el logger podría no haberse importado.
    print(f"ERROR CRÍTICO: Falta un archivo del proyecto. No se puede iniciar. Detalle: {e}")
    # Mostramos un diálogo de error simple si es posible
    app = QApplication(sys.argv)
    QMessageBox.critical(
        None,
        "Error de Archivos del Proyecto",
        f"Falta un archivo esencial para ejecutar la aplicación.\n\n"
        f"Por favor, asegúrese de que todos los archivos .py estén en el directorio correcto.\n\n"
        f"Error: {e}"
    )
    sys.exit(1)

def main():
    """
    Punto de entrada principal para iniciar la aplicación Copiloto-ID.
    """
    logger.info("=============================================")
    logger.info("      Iniciando aplicación Copiloto-ID       ")
    logger.info("=============================================")

    # Crear la aplicación Qt ANTES de cualquier diálogo de error para que sean visibles.
    app = QApplication(sys.argv)

    # --- Verificaciones Críticas Antes de Lanzar la GUI ---
    try:
        # 1. Asegurar que los directorios base existan
        required_dirs = [
            "models", 
            "database", 
            "rostros_capturados", 
            "logs", 
            "logs/data_history", 
            "logs/sleep_events", 
            "logs/hdr_images",
            "temp",
            "config"
        ]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        logger.info("Directorios del proyecto verificados/creados.")

        # 2. Configurar/verificar el archivo config.json
        is_first_run = ConfigManager.setup_config_file()
        logger.info("Gestor de configuración inicializado.")

        # 3. Inicializar/verificar explícitamente la base de datos y su esquema
        DatabaseManager.initialize_database()
        logger.info("Gestor de base de datos inicializado.")

    except Exception as e:
        logger.critical(f"Fallo crítico durante la inicialización del sistema: {e}", exc_info=True)
        QMessageBox.critical(
            None, 
            "Error de Inicialización",
            f"No se pudo iniciar la aplicación debido a un error crítico.\n\n"
            f"Detalle: {e}\n\n"
            "Por favor, revise el archivo de log para más detalles."
        )
        sys.exit(1)
        
    # --- Lanzamiento de la Ventana Principal ---
    try:
        window = MainWindow()
        window.show()
        
        # Si setup_config_file() determinó que es la primera ejecución,
        # le ordenamos a la ventana principal que inicie el diálogo de creación de clave.
        if is_first_run:
            logger.info("Detectada primera ejecución. Solicitando creación de clave de administrador.")
            QTimer.singleShot(100, window.trigger_first_run_setup)

        logger.info("MainWindow creada y mostrada. La aplicación está en funcionamiento.")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"Excepción no controlada en el nivel principal de la aplicación: {e}", exc_info=True)
        QMessageBox.critical(
            None,
            "Error Inesperado",
            f"La aplicación ha encontrado un error fatal y debe cerrarse.\n\n"
            f"Detalle: {e}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()