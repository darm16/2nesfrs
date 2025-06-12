# modules/behavioral/notificaciones.py

import queue
import subprocess
import os
from typing import Optional
import csv
import threading
from datetime import datetime
from collections import defaultdict
from typing import Dict, List
from logging_setup import logger
from config_manager import ConfigManager

# --- Variables de estado del Módulo ---

# Cola de prioridad para manejar los mensajes de voz. Formato: (prioridad, texto)
# Un número de prioridad más bajo significa mayor urgencia.
voice_queue = queue.PriorityQueue()

# Flag para evitar que se reproduzcan múltiples audios simultáneamente
is_speaking_lock = threading.Lock()
is_speaking = False

# Estructuras en memoria para el conteo y la duración de eventos durante la sesión
event_counts = defaultdict(int)
event_durations = defaultdict(float)
event_log_session = []
program_start_time = datetime.now()


def log_event(event_name: str, duration: float = None):
    """
    Registra un evento en las estructuras de datos en memoria para la sesión actual.

    Args:
        event_name (str): El nombre del evento (ej. "Bostezar", "Cabeceo").
        duration (float, optional): La duración del evento en segundos, si aplica.
    """
    global event_counts, event_durations, event_log_session
    timestamp = datetime.now()
    event_log_session.append((timestamp, event_name, duration))
    event_counts[event_name] += 1
    if duration:
        event_durations[event_name] += duration
    logger.info(f"Evento en memoria registrado: {event_name}")

def speak(event_key: str):
    """
    Añade una alerta de voz a la cola de prioridad, leyendo su configuración desde config.json.

    Args:
        event_key (str): La clave del evento que corresponde a una entrada en la sección
                         'voice_alerts' del config.json (ej. "Despierta", "Distraccion").
    """
    try:
        config = ConfigManager.load_full_config()
        alerts_config = config.get('voice_alerts', {})
        
        event_config = alerts_config.get(event_key)
        
        if not event_config:
            logger.warning(f"No se encontró configuración de alerta de voz para la clave: '{event_key}'")
            return

        if event_config.get('enabled', False):
            text_to_speak = event_config.get('text', '')
            priority = event_config.get('priority', 3) # Default a baja prioridad
            
            if text_to_speak:
                # Limpiar la cola de alertas de menor prioridad si llega una crítica
                if priority < 2 and not voice_queue.empty():
                    # Esta lógica podría ser más compleja, por ahora es simple
                    pass
                
                voice_queue.put((priority, text_to_speak))
                logger.info(f"Alerta '{event_key}' con prioridad {priority} añadida a la cola de voz.")
        
    except Exception as e:
        logger.error(f"Error al procesar la solicitud de voz para '{event_key}': {e}")

def _run_command(command: List[str]):
    """Ejecuta un comando de sistema y maneja errores."""
    try:
        # Usamos DEVNULL para suprimir la salida de los comandos en la consola
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        logger.error(f"Error: El comando '{command[0]}' no se encontró. Asegúrese de que esté instalado y en el PATH del sistema.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al ejecutar el comando '{' '.join(command)}': {e}")
        return False
    except Exception as e:
        logger.error(f"Excepción inesperada al ejecutar comando: {e}")
        return False

def voice_handler():
    """
    Función que se ejecuta en un hilo separado. Procesa la cola de voz de forma continua.
    """
    global is_speaking
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    output_wav = os.path.join(temp_dir, 'output.wav')
    
    while True:
        try:
            priority, text = voice_queue.get()
            
            with is_speaking_lock:
                is_speaking = True

            logger.info(f"Reproduciendo alerta (Prioridad {priority}): '{text}'")
            
            # 1. Sintetizar texto a archivo WAV usando pico2wave
            if not _run_command(['pico2wave', '-l', 'es-ES', '-w', output_wav, text]):
                # Si pico2wave falla, no podemos continuar con este item
                with is_speaking_lock:
                    is_speaking = False
                voice_queue.task_done()
                continue

            # 2. Reproducir el archivo WAV usando aplay
            _run_command(['aplay', '-q', output_wav]) # -q para modo silencioso
                
            # Limpiar archivo temporal
            if os.path.exists(output_wav):
                os.remove(output_wav)

        except queue.Empty:
            time.sleep(0.1) # Esperar si la cola está vacía
        except Exception as e:
            logger.error(f"Error en el manejador de voz (voice_handler): {e}")
        finally:
            with is_speaking_lock:
                is_speaking = False
            voice_queue.task_done()


# --- Funciones de utilidad para acceder a los datos de la sesión ---

def get_session_event_counts() -> Dict[str, int]:
    """Devuelve una copia de los contadores de eventos de la sesión actual."""
    return dict(event_counts)

def reset_session_data():
    """Reinicia todos los contadores y logs para una nueva sesión de monitoreo."""
    global event_counts, event_durations, event_log_session, program_start_time
    event_counts.clear()
    event_durations.clear()
    event_log_session.clear()
    program_start_time = datetime.now()
    logger.info("Datos de sesión de notificaciones reseteados.")

def save_event_log_to_file(user_code: Optional[str] = None):
    """
    Guarda el historial de eventos de la sesión actual en un archivo CSV.
    """
    if not event_log_session:
        logger.info("No hay eventos para guardar en el registro de sesión.")
        return

    # Crear directorio de logs si no existe
    log_dir = os.path.join("logs", "session_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Construir un nombre de archivo más descriptivo si se proporciona el código de usuario
    user_part = f"{user_code}_" if user_code else ""
    filename = os.path.join(log_dir, f"event_log_{user_part}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Escribir encabezado
            writer.writerow(["timestamp", "event_name", "duration_seconds"])
            
            # Escribir cada evento
            for timestamp, event_name, duration in event_log_session:
                writer.writerow([
                    timestamp.isoformat(timespec='seconds'),
                    event_name,
                    f"{duration:.2f}" if duration is not None else ""
                ])
        
        logger.info(f"Historial de sesión guardado en {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error al guardar el historial de eventos: {e}")
        return None

def save_events_on_final_exit():
    """Guarda el log final al cerrar el programa."""
    save_event_log_to_file(user_code="CIERRE_PROGRAMA")

# Iniciar el hilo manejador de voz al importar el módulo
# Se configura como 'daemon' para que no bloquee el cierre de la aplicación principal
voice_thread = threading.Thread(target=voice_handler, daemon=True)
voice_thread.start()

# Registrar la función de guardado al finalizar el programa
import atexit
atexit.register(save_events_on_final_exit)