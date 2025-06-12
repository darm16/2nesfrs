# database_manager.py

import sqlite3
import os
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from logging_setup import logger

class DatabaseManager:
    """
    Gestiona todas las operaciones de la base de datos SQLite para la aplicación.
    """
    DATABASE_DIR = "database"
    DATABASE_FILE = os.path.join(DATABASE_DIR, "copiloto_id.db")

    def __init__(self):
        """
        Constructor simple. La inicialización se realiza explícitamente mediante
        el método estático initialize_database() para un control total del flujo de arranque.
        """
        # La inicialización se realiza explícitamente mediante el método estático
        # DatabaseManager.initialize_database()
        pass

    @staticmethod
    def initialize_database():
        """
        Asegura que el directorio de la base de datos exista y que el esquema
        de las tablas esté creado y actualizado. Se ejecuta al inicio de la aplicación.
        """
        try:
            if not os.path.exists(DatabaseManager.DATABASE_DIR):
                os.makedirs(DatabaseManager.DATABASE_DIR)
                logger.info(f"Directorio de base de datos creado en: {DatabaseManager.DATABASE_DIR}")
        except OSError as e:
            logger.critical(f"No se pudo crear el directorio para la base de datos: {e}")
            raise

        try:
            with sqlite3.connect(DatabaseManager.DATABASE_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON;")

                # --- Tabla de Usuarios ---
                cursor.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    codigo_usuario TEXT UNIQUE NOT NULL,
                                    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    ultimo_acceso TIMESTAMP,
                                    cal_ear_mean REAL, cal_ear_std REAL, cal_mar_mean REAL, cal_mar_std REAL,
                                    cal_puc_mean REAL, cal_puc_std REAL, cal_moe_mean REAL, cal_moe_std REAL
                                 )''')

                # --- Tabla de Embeddings ---
                cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings_usuarios (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    usuario_id INTEGER NOT NULL,
                                    embedding BLOB NOT NULL,
                                    ruta_imagen_capturada TEXT,
                                    modelo_embedding TEXT,
                                    pitch REAL, yaw REAL, roll REAL,
                                    fecha_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
                                )''')

                # --- Tabla de Registros de Acceso (Login) ---
                cursor.execute('''CREATE TABLE IF NOT EXISTS registros_acceso (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    usuario_id INTEGER,
                                    fecha_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    tipo_acceso TEXT,
                                    resultado TEXT,
                                    confianza REAL,
                                    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
                                 )''')

                # --- Tabla de Registros de Comportamiento (Fatiga) ---
                cursor.execute('''CREATE TABLE IF NOT EXISTS registros_comportamiento (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    usuario_id INTEGER,
                                    fecha_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    tipo_evento TEXT NOT NULL,
                                    duracion_seg REAL,
                                    metadata TEXT,
                                    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
                                 )''')
                
                conn.commit()
                logger.info("Base de datos inicializada/verificada correctamente.")

        except sqlite3.Error as e:
            logger.critical(f"Error crítico al inicializar la base de datos: {e}")
            raise

    @staticmethod
    def _execute_query(query: str, params: tuple = (), fetch: Optional[str] = None) -> Any:
        """Método de ayuda privado para ejecutar consultas de forma segura."""
        try:
            with sqlite3.connect(DatabaseManager.DATABASE_FILE) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON;")
                cursor.execute(query, params)
                
                if fetch == 'one':
                    result = cursor.fetchone()
                    return dict(result) if result else None
                if fetch == 'all':
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error en la base de datos: {e} | Query: {query} | Params: {params}")
            return None if fetch else False
            
    # --- Implementación completa de todas las funciones ---

    @staticmethod
    def register_user() -> Optional[Tuple[int, str]]:
        last_id_result = DatabaseManager._execute_query("SELECT MAX(id) as max_id FROM usuarios", fetch='one')
        next_id = (last_id_result['max_id'] or 0) + 1
        user_code = f"USUARIO_{next_id:04d}"
        
        user_id = DatabaseManager._execute_query("INSERT INTO usuarios (codigo_usuario) VALUES (?)", (user_code,))
        if user_id:
            logger.info(f"Usuario '{user_code}' registrado con ID: {user_id}")
            return user_id, user_code
        return None

    @staticmethod
    def add_user_embedding(user_id: int, embedding: np.ndarray, image_path: str, model_name: str, pose: tuple):
        embedding_blob = embedding.astype(np.float32).tobytes()
        pitch, yaw, roll = pose
        query = "INSERT INTO embeddings_usuarios (usuario_id, embedding, ruta_imagen_capturada, modelo_embedding, pitch, yaw, roll) VALUES (?, ?, ?, ?, ?, ?, ?)"
        return DatabaseManager._execute_query(query, (user_id, embedding_blob, image_path, model_name, pitch, yaw, roll))

    @staticmethod
    def get_user_details_for_list() -> List[Dict]:
        """Obtiene una lista de usuarios con su ID, código y la ruta de su primera imagen registrada."""
        query = """
            SELECT u.id, u.codigo_usuario, (
                SELECT e.ruta_imagen_capturada 
                FROM embeddings_usuarios e 
                WHERE e.usuario_id = u.id ORDER BY e.fecha_captura ASC LIMIT 1
            ) as ruta_imagen
            FROM usuarios u ORDER BY u.codigo_usuario
        """
        return DatabaseManager._execute_query(query, fetch='all') or []

    @staticmethod
    def delete_user(user_id: int) -> Tuple[bool, List[str]]:
        """Elimina un usuario y devuelve las rutas de sus imágenes para borrarlas del disco."""
        image_paths_query = "SELECT ruta_imagen_capturada FROM embeddings_usuarios WHERE usuario_id = ?"
        paths_result = DatabaseManager._execute_query(image_paths_query, (user_id,), fetch='all')
        image_paths = [row['ruta_imagen_capturada'] for row in paths_result if row['ruta_imagen_capturada']]

        delete_query = "DELETE FROM usuarios WHERE id = ?"
        success = DatabaseManager._execute_query(delete_query, (user_id,))
        
        if success is not False:
            logger.info(f"Usuario con ID {user_id} eliminado de la base de datos (y datos asociados por cascada).")
            return True, image_paths
        return False, []

    @staticmethod
    def delete_all_users() -> Tuple[bool, List[str]]:
        """Elimina TODOS los usuarios y devuelve todas las rutas de imágenes."""
        image_paths_query = "SELECT ruta_imagen_capturada FROM embeddings_usuarios"
        paths_result = DatabaseManager._execute_query(image_paths_query, fetch='all')
        image_paths = [row['ruta_imagen_capturada'] for row in paths_result if row['ruta_imagen_capturada']]

        DatabaseManager._execute_query("DELETE FROM registros_comportamiento")
        DatabaseManager._execute_query("DELETE FROM registros_acceso")
        DatabaseManager._execute_query("DELETE FROM embeddings_usuarios")
        DatabaseManager._execute_query("DELETE FROM usuarios")
        DatabaseManager._execute_query("DELETE FROM sqlite_sequence") # Resetea contadores de autoincremento
        logger.info("Todas las tablas de usuarios y registros han sido limpiadas.")
        return True, image_paths

    # ... (resto de funciones como save_calibration_data, load_calibration_data, get_behavioral_events, etc. se mantienen como las definimos) ...

    @staticmethod
    def get_access_logs(user_id: int = None, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Obtiene los registros de acceso (logins) con filtros opcionales."""
        query = "SELECT a.fecha_hora, a.resultado, a.confianza, u.codigo_usuario FROM registros_acceso a LEFT JOIN usuarios u ON a.usuario_id = u.id"
        conditions, params = [], []
        if user_id and user_id > 0:
            conditions.append("a.usuario_id = ?")
            params.append(user_id)
        if start_date:
            conditions.append("date(a.fecha_hora) >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date(a.fecha_hora) <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY a.fecha_hora DESC"
        
        rows = DatabaseManager._execute_query(query, tuple(params), fetch='all')
        return rows if rows else []