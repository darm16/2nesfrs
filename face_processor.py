# face_processor.py

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Any, Optional, List, Dict

from logging_setup import logger
from config_manager import ConfigManager

class EmbeddingModel:
    """Clase interna para cargar y usar el modelo de embedding facial ONNX."""
    def __init__(self, config: Dict):
        self.initialized = False
        self.ort_session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        
        rec_settings = config.get('recognition_settings', {})
        model_filename = rec_settings.get('model_filename')
        model_path = Path(__file__).parent / "models" / model_filename if model_filename else None
        logger.info(f"Buscando modelo en ruta: {model_path}")
        logger.info(f"Ruta absoluta: {model_path.absolute() if model_path else 'N/A'}")
        logger.info(f"¿Existe? {model_path.exists() if model_path else 'N/A'}")

        if not model_path or not model_path.exists():
            logger.critical(f"El archivo del modelo '{model_filename}' no se encuentra en la ruta: {model_path.absolute() if model_path else 'N/A'}")
            logger.critical(f"Directorio actual: {Path.cwd()}")
            return

        try:
            # Seleccionar proveedor de ejecución (GPU si está disponible, si no CPU)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            
            self.ort_session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name

            logger.info(f"Modelo ONNX '{model_filename}' cargado exitosamente usando {providers[0]}.")
            self.initialized = True

        except Exception as e:
            logger.critical(f"Error fatal al cargar el modelo ONNX '{model_filename}': {e}", exc_info=True)
            self.initialized = False

    def get_face_embedding(self, preprocessed_blob: np.ndarray) -> Optional[np.ndarray]:
        """Genera el vector de embedding a partir de un blob preprocesado."""
        if not self.initialized or self.ort_session is None:
            return None
        try:
            embedding_raw = self.ort_session.run([self.output_name], {self.input_name: preprocessed_blob})[0]
            embedding = np.array(embedding_raw, dtype=np.float32).flatten()
            
            # Normalizar el vector de embedding (norma L2)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 1e-6 else embedding
        except Exception as e:
            logger.error(f"Error al generar embedding con ONNX Runtime: {e}", exc_info=True)
            return None

class FaceProcessor:
    """Orquesta el procesamiento facial: detección, alineación y extracción de embeddings."""
    def __init__(self, config: Dict):
        self.config = config
        self.initialized = False
        
        try:
            # Inicializar Face Mesh
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Inicializar detección de manos
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.5
            )
            
            # Inicializar detección de postura
            self.pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.2,
                min_tracking_confidence=0.5
            )
            
            # Utilidades de dibujo
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info("MediaPipe (Face, Hands, Pose) inicializado.")
            
            self.embedding_model = EmbeddingModel(config)
            if not self.embedding_model.initialized:
                raise RuntimeError("El modelo de embedding no se pudo inicializar.")

            self.initialized = True
            logger.info("FaceProcessor inicializado correctamente con soporte para manos y postura.")

        except Exception as e:
            logger.critical(f"Error al inicializar FaceProcessor: {e}", exc_info=True)
            self.initialized = False
            
    def process_frame_for_landmarks(self, frame_bgr: np.ndarray) -> Optional[Any]:
        """Procesa un frame BGR para detectar landmarks faciales."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Optimización: pasar el frame como de solo lectura
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def draw_face_mesh(self, frame: np.ndarray, face_landmarks: Any) -> np.ndarray:
        """Dibuja la malla facial sobre un frame para visualización."""
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        return frame

    def align_face(self, frame_bgr: np.ndarray, face_landmarks: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extrae y alinea un "chip" facial a partir de los landmarks.
        Devuelve el blob preprocesado para el modelo y la imagen alineada para guardar.
        """
        h, w, _ = frame_bgr.shape
        lm = face_landmarks.landmark

        try:
            # Puntos de referencia estándar para la alineación facial
            src_pts = np.array([
                (lm[33].x*w, lm[33].y*h),   # Ojo izquierdo, esquina izq.
                (lm[263].x*w, lm[263].y*h), # Ojo derecho, esquina der.
                (lm[1].x*w, lm[1].y*h),     # Punta de la nariz
                (lm[61].x*w, lm[61].y*h),    # Comisura de la boca izq.
                (lm[291].x*w, lm[291].y*h)   # Comisura de la boca der.
            ], dtype=np.float32)
        except IndexError:
            logger.error("Error de índice al acceder a landmarks para alinear rostro.")
            return None

        # Puntos de destino estándar para un chip de 112x112
        dst_pts = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014],
            [48.0252, 71.7366], [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)

        # Calcular la matriz de transformación afín
        transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if transform_matrix is None:
            logger.warning("No se pudo estimar la transformación afín para alinear el rostro.")
            return None

        # Aplicar la transformación para obtener la cara alineada
        input_size = (112, 112) # El tamaño que espera el modelo ONNX
        aligned_face = cv2.warpAffine(frame_bgr, transform_matrix, input_size)
        
        # Preprocesar la imagen para el modelo ONNX (normalización BGR, rango -1 a 1)
        blob = cv2.dnn.blobFromImage(
            aligned_face, scalefactor=1./127.5, size=input_size, mean=(127.5, 127.5, 127.5), swapRB=True
        )
        return blob, aligned_face

    def get_embedding(self, aligned_blob: np.ndarray) -> Optional[np.ndarray]:
        """Genera el embedding a partir de un blob alineado."""
        return self.embedding_model.get_face_embedding(aligned_blob)
        
    @staticmethod
    def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compara dos embeddings usando similitud de coseno."""
        if emb1 is None or emb2 is None: return 0.0
        return float(np.dot(emb1, emb2))

    def find_match(self, query_emb: np.ndarray, known_embs_data: List[Dict], threshold: float) -> Tuple[Optional[Dict], float]:
        """
        Busca el rostro más similar en una lista de embeddings conocidos.

        Args:
            query_emb: El embedding del rostro a buscar.
            known_embs_data: Lista de diccionarios, cada uno con 'user_id', 'codigo_usuario', y 'embedding'.
            threshold: El umbral de similitud para considerar una coincidencia.

        Returns:
            Una tupla (diccionario_del_usuario_coincidente, similitud_máxima).
        """
        if query_emb is None or not known_embs_data:
            return None, 0.0

        best_match_user_data = None
        highest_similarity = -1.0

        for user_data in known_embs_data:
            similarity = self.compare_embeddings(query_emb, user_data['embedding'])
            if similarity > highest_similarity:
                highest_similarity = similarity
                if similarity >= threshold:
                    best_match_user_data = user_data
        
        return best_match_user_data, highest_similarity


    def close(self):
        """Libera los recursos de MediaPipe."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            logger.info("Recursos de MediaPipe Face Mesh liberados.")
            
        if hasattr(self, 'hands'):
            self.hands.close()
            logger.info("Recursos de MediaPipe Hands liberados.")
            
        if hasattr(self, 'pose'):
            self.pose.close()
            logger.info("Recursos de MediaPipe Pose liberados.")