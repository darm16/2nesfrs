# gui.py

import sys
import os
import cv2
import numpy as np
import time
import copy
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFrame, QGroupBox, QTreeWidget, QTreeWidgetItem, 
                             QHeaderView, QListWidget, QListWidgetItem, QTextEdit, QComboBox, 
                             QSlider, QMessageBox, QTabWidget, QDateEdit, QTableWidget, QTableWidgetItem,
                             QInputDialog, QLineEdit, QSizePolicy, QFormLayout, QMenu)
from PyQt5.QtCore import Qt, QTimer, QSize, QDate, pyqtSignal, QPoint, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# --- Módulos del Proyecto ---
from logging_setup import logger
from config_manager import ConfigManager
from database_manager import DatabaseManager
from camera_thread import CameraThread
from face_processor import FaceProcessor
from pose_estimator import PoseEstimator
from fatigue_processor import FatigueProcessor
from settings_dialog import SettingsDialog
from analytics_processor import AnalyticsProcessor
import roi_autoexp
import notificaciones
import mediapipe as mp  # Importación directa de mediapipe

try:
    from mpu_thread import MPUThread
except ImportError:
    MPUThread = None
    logger.warning("No se pudo importar MPUThread. El modo reposo por hardware no estará disponible.")

# --- Hilo para Tareas Pesadas en Segundo Plano ---
class AnalyticsWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(Exception)

    def __init__(self, db_manager, user_id, start_date, end_date):
        super().__init__()
        self.db_manager = db_manager
        self.user_id = user_id
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            logger.info(f"Worker de analíticas iniciado para usuario ID: {self.user_id}...")
            events_data = self.db_manager.get_behavioral_events(self.user_id, self.start_date, self.end_date)
            processor = AnalyticsProcessor(events_data)
            kpis = processor.calculate_kpis()
            result = {'kpis': kpis, 'events_data': events_data}
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Error en el worker de analíticas: {e}", exc_info=True)
            self.error.emit(e)

# --- Clase Personalizada para la Lista de Usuarios ---
class UserImageLabel(QLabel):
    deleteUser = pyqtSignal(int)
    viewUserDetails = pyqtSignal(int)

    def __init__(self, user_id: int, user_code: str, image_path: str, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.user_code = user_code
        self.image_path = image_path
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, position: QPoint):
        context_menu = QMenu(self)
        view_action = context_menu.addAction("Ver Detalles")
        delete_action = context_menu.addAction("Eliminar Usuario")
        action = context_menu.exec_(self.mapToGlobal(position))
        if action == view_action:
            self.viewUserDetails.emit(self.user_id)
        elif action == delete_action:
            self.deleteUser.emit(self.user_id)

# --- Clase Principal de la GUI ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Copiloto-ID - Sistema de Monitoreo Integral v1.0")
        self.setGeometry(50, 50, 1400, 900)
        self.setStyleSheet("background-color: #2E2E2E; color: #E0E0E0; font-family: Segoe UI;")
        if os.path.exists('icon.png'): self.setWindowIcon(QIcon('icon.png')) 

        logger.info("Inicializando MainWindow...")
        
        # Inicializar atributos importantes primero
        self.camera_status = QLabel("Estado: Inicializando...")
        self.status_label = QLabel("Iniciando...")
        
        self.config = ConfigManager.load_full_config()
        self.db_manager = DatabaseManager()
        self.face_processor = FaceProcessor(self.config)
        self.fatigue_processor = FatigueProcessor(self.config)
        
        if not self.face_processor.initialized:
            QMessageBox.critical(self, "Error Crítico", "No se pudo inicializar el procesador facial.")
            sys.exit(1)

        self.camera_thread: Optional[CameraThread] = None
        self.mpu_thread: Optional[MPUThread] = None
        self.analytics_worker_thread: Optional[QThread] = None
        
        # Variables para el manejo de cámaras
        self.capturing = False
        max_idx_to_check = self.config['camera_settings'].get('MAX_CAMERA_INDEX_TO_CHECK', 4)
        self.available_cameras = self._detect_available_cameras(max_idx_to_check)
        
        # Obtener el índice de la cámara actual del config, o la primera disponible
        self.camera_index = self.config['camera_settings'].get('camera_index', 0)
        if self.available_cameras and self.camera_index not in self.available_cameras:
            self.camera_index = self.available_cameras[0]
            self.config['camera_settings']['camera_index'] = self.camera_index
            ConfigManager.save_full_config(self.config)
        
        # Si no hay cámaras, establecer el índice a -1
        if not self.available_cameras:
            self.camera_index = -1
            
        self.app_state = "LISTENING"
        self.current_user_id: Optional[int] = None
        self.current_user_code: str = "N/A"
        self.current_frame_array: Optional[np.ndarray] = None
        self.last_monitoring_data: Dict = {}
        self.video_is_visible = True
        self.photos_are_visible = not self.config.get('privacy_settings', {}).get('privacy_mode_default', True)
        self.fatigue_event_counts = defaultdict(int)
        self.is_calibrating = False
        self.calibration_frames_data = []

        self.state_timers = {
            'stable_face_counter': 0, 'face_lost_counter': 0,
            'auto_register_start_time': None, 'enrichment_start_time': None,
            'session_embeddings_captured': 0, 'session_captured_poses': []
        }

        self.initUI()
        self.load_users_to_list()
        
        QTimer.singleShot(100, self.trigger_first_run_setup_if_needed)

        # Iniciar la cámara después de que la interfaz esté lista
        QTimer.singleShot(100, self._initialize_camera_and_mpu)
        
    def _initialize_camera_and_mpu(self):
        """Inicializa la cámara y el MPU después de que la interfaz esté lista."""
        if self.config.get('hardware_settings', {}).get('USE_MPU', False) and MPUThread:
            self._start_mpu_thread()
        else:
            logger.info("Modo reposo por MPU desactivado.")
        
        # Iniciar la cámara si hay una disponible
        if self.camera_index != -1:
            self.start_camera()
        else:
            logger.warning("No hay cámaras disponibles para iniciar.")
            self.camera_status.setText("Estado: Sin cámaras disponibles")
            self.camera_status.setStyleSheet("color: red;")

    # --- MÉTODOS DE CONSTRUCCIÓN DE LA INTERFAZ ---
    def initUI(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        left_panel = self._create_left_panel(); left_panel.setFixedWidth(380)
        right_panel = self._create_right_panel()
        main_layout.addWidget(left_panel); main_layout.addWidget(right_panel, 1)

    def _detect_available_cameras(self, max_idx_to_check: int) -> list[int]:
        """Detecta los índices de las cámaras disponibles."""
        available_indices = []
        logger.info(f"Detectando cámaras hasta índice {max_idx_to_check}...")
        
        for i in range(max_idx_to_check + 1):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                # Intenta leer un frame para confirmar que es funcional
                ret, _ = cap.read()
                if ret:
                    logger.info(f"Cámara funcional en índice {i}.")
                    available_indices.append(i)
                cap.release()  # Liberar el recurso inmediatamente
            # Si cap no se abrió, no es necesario llamar a .release()
            
        if not available_indices:
            logger.warning("No se detectaron cámaras funcionales.")
        else:
            logger.info(f"Cámaras funcionales detectadas: {available_indices}")
            
        return available_indices
        
    def _create_left_panel(self) -> QWidget:
        panel = QWidget(); layout = QVBoxLayout(panel); layout.setSpacing(15)
        title_label = QLabel("Copiloto-ID"); title_label.setFont(QFont("Segoe UI", 18, QFont.Bold)); title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Grupo de controles de cámara
        camera_group = QGroupBox("Configuración de Cámara")
        camera_layout = QVBoxLayout()
        
        # Selector de cámara
        camera_selector_layout = QHBoxLayout()
        camera_selector_layout.addWidget(QLabel("Cámara:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("Seleccione la cámara a utilizar")
        self._update_camera_selector()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_selector_layout.addWidget(self.camera_combo)
        
        # Botón de actualizar
        self.refresh_button = QPushButton("Actualizar")
        self.refresh_button.setToolTip("Actualizar lista de cámaras disponibles")
        self.refresh_button.clicked.connect(self._refresh_camera_list)
        camera_selector_layout.addWidget(self.refresh_button)
        
        camera_layout.addLayout(camera_selector_layout)
        camera_group.setLayout(camera_layout)
        
        # Estado de la cámara (ya inicializado en __init__)
        self.camera_status.setText("Estado: Desconocido")
        camera_layout.addWidget(self.camera_status)
        
        layout.addWidget(camera_group)
        
        # Estado del sistema (ya inicializado en __init__)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 5px; background-color: #3C3C3C; border: 1px solid #555; border-radius: 5px; min-height: 40px;")
        layout.addWidget(self.status_label)
        users_group = QGroupBox("Gestión de Usuarios"); users_layout = QVBoxLayout(users_group)
        self.user_list_widget = QListWidget(); self.user_list_widget.setStyleSheet("QListWidget::item { padding: 5px; border-bottom: 1px solid #3C3C3C; }")
        users_layout.addWidget(self.user_list_widget, 1)
        user_buttons_layout = QHBoxLayout()
        self.toggle_photos_button = QPushButton("Mostrar Fotos"); self.toggle_photos_button.setCheckable(True); self.toggle_photos_button.setChecked(self.photos_are_visible)
        self.toggle_photos_button.setToolTip("Muestra u oculta las fotos de los usuarios.\nRequiere clave si el modo privacidad está activo."); self.toggle_photos_button.clicked.connect(self._toggle_photo_visibility)
        self.settings_button = QPushButton("Configuración"); self.settings_button.setToolTip("Abre la ventana de configuración del sistema (requiere clave)."); self.settings_button.clicked.connect(self._open_settings_dialog)
        user_buttons_layout.addWidget(self.toggle_photos_button); user_buttons_layout.addWidget(self.settings_button)
        users_layout.addLayout(user_buttons_layout)
        self.toggle_video_button = QPushButton("Ocultar Vídeo"); self.toggle_video_button.setCheckable(True); self.toggle_video_button.setToolTip("Oculta el stream de vídeo para ahorrar recursos.\nLa detección seguirá funcionando en segundo plano.")
        self.toggle_video_button.toggled.connect(self._toggle_video_visibility_action); users_layout.addWidget(self.toggle_video_button)
        layout.addWidget(users_group, 1)
        return panel

    def _create_right_panel(self) -> QTabWidget:
        self.tabs = QTabWidget()
        monitoring_tab = QWidget(); monitoring_layout = QVBoxLayout(monitoring_tab)
        self.camera_label = QLabel("Iniciando Cámara..."); self.camera_label.setAlignment(Qt.AlignCenter); self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 5px;")
        monitoring_layout.addWidget(self.camera_label, 3)
        fatigue_panel = self._create_fatigue_panel(); monitoring_layout.addWidget(fatigue_panel, 1)
        self.tabs.addTab(monitoring_tab, "Monitoreo en Vivo")
        analytics_tab = self._create_analytics_tab(); self.tabs.addTab(analytics_tab, "Analíticas y Reportes")
        return self.tabs
        
    def _create_fatigue_panel(self) -> QFrame:
        panel = QFrame(); panel.setFrameShape(QFrame.NoFrame); layout = QHBoxLayout(panel); layout.setContentsMargins(0, 5, 0, 0)
        metrics_group = QGroupBox("Métricas en Tiempo Real"); metrics_layout = QFormLayout(metrics_group)
        self.ear_label = QLabel("N/A"); self.mar_label = QLabel("N/A"); self.puc_label = QLabel("N/A"); self.moe_label = QLabel("N/A")
        metrics_layout.addRow("EAR (Ojos):", self.ear_label); metrics_layout.addRow("MAR (Boca):", self.mar_label)
        metrics_layout.addRow("PUC (Pupila):", self.puc_label); metrics_layout.addRow("MOE (Ratio):", self.moe_label)
        log_group = QGroupBox("Registro de Eventos de Sesión"); log_layout = QVBoxLayout(log_group)
        self.fatigue_event_tree = QTreeWidget(); self.fatigue_event_tree.setColumnCount(3); self.fatigue_event_tree.setHeaderLabels(["Evento", "Habilitado", "Cantidad"])
        self.fatigue_event_tree.header().setSectionResizeMode(0, QHeaderView.Stretch); self.fatigue_event_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.fatigue_event_tree.itemDoubleClicked.connect(self._toggle_event_status); log_layout.addWidget(self.fatigue_event_tree)
        self._populate_event_tree()
        layout.addWidget(metrics_group, 1); layout.addWidget(log_group, 2)
        return panel
    
    def _create_analytics_tab(self) -> QWidget:
        analytics_widget = QWidget(); main_layout = QVBoxLayout(analytics_widget)
        filters_group = QGroupBox("Filtros del Reporte"); filters_layout = QHBoxLayout(filters_group)
        filters_layout.addWidget(QLabel("Usuario:")); self.analytics_user_combo = QComboBox(); self.analytics_user_combo.setMinimumWidth(150)
        self.analytics_user_combo.setToolTip("Seleccione un usuario para ver su historial."); filters_layout.addWidget(self.analytics_user_combo)
        filters_layout.addWidget(QLabel("Desde:")); self.analytics_start_date = QDateEdit(QDate.currentDate().addMonths(-1)); self.analytics_start_date.setCalendarPopup(True)
        self.analytics_start_date.setDisplayFormat("yyyy-MM-dd"); filters_layout.addWidget(self.analytics_start_date)
        filters_layout.addWidget(QLabel("Hasta:")); self.analytics_end_date = QDateEdit(QDate.currentDate()); self.analytics_end_date.setCalendarPopup(True)
        self.analytics_end_date.setDisplayFormat("yyyy-MM-dd"); filters_layout.addWidget(self.analytics_end_date)
        filters_layout.addStretch()
        self.generate_report_button = QPushButton("Generar Reporte"); self.generate_report_button.clicked.connect(self._handle_generate_report)
        filters_layout.addWidget(self.generate_report_button); main_layout.addWidget(filters_group)
        kpi_group = QGroupBox("Resumen del Reporte (KPIs)"); kpi_layout = QFormLayout(kpi_group)
        self.kpi_total_events_label = QLabel("N/A"); self.kpi_most_frequent_label = QLabel("N/A"); self.kpi_peak_hour_label = QLabel("N/A")
        kpi_layout.addRow("Total de Eventos:", self.kpi_total_events_label); kpi_layout.addRow("Evento Más Frecuente:", self.kpi_most_frequent_label)
        kpi_layout.addRow("Hora Pico de Fatiga:", self.kpi_peak_hour_label); main_layout.addWidget(kpi_group)
        log_group = QGroupBox("Historial de Eventos Detallado"); log_layout = QVBoxLayout(log_group)
        self.report_table = QTableWidget(); self.report_table.setColumnCount(4); self.report_table.setHorizontalHeaderLabels(["Fecha y Hora", "Usuario", "Tipo de Evento", "Detalles"])
        self.report_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents); self.report_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers); self.report_table.setSelectionBehavior(QTableWidget.SelectRows)
        log_layout.addWidget(self.report_table); main_layout.addWidget(log_group, 1)
        return analytics_widget

    def trigger_first_run_setup_if_needed(self):
        """Verifica si es la primera ejecución para forzar la creación de una clave."""
        is_first_run = not self.config['privacy_settings']['photo_view_password_hash']
        if is_first_run:
            QMessageBox.information(self, "Configuración Inicial Requerida", "Bienvenido a Copiloto-ID.\n\nComo es la primera ejecución, debe establecer una clave de administrador.")
            self._handle_change_password(is_first_run=True)

    # --- Lógica de la aplicación completa, incluyendo todos los manejadores de estado y eventos ---
    # ... (La segunda y última parte del código irá en la siguiente respuesta) ...
    # gui.py (Parte 2 de 6 - Añadir estos métodos a la clase MainWindow)

    # --- MÉTODOS DE CONSTRUCCIÓN DE LA INTERFAZ ---

    def initUI(self):
        """
        Construye la estructura principal de la interfaz gráfica, organizando
        los paneles izquierdo y derecho.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        left_panel = self._create_left_panel()
        left_panel.setFixedWidth(380)
        
        right_panel = self._create_right_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1) # El panel derecho es expandible

    def _create_left_panel(self) -> QWidget:
        """Crea el panel izquierdo completo con todos sus controles."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        title_label = QLabel("Copiloto-ID")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.status_label = QLabel("Iniciando...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 5px; background-color: #3C3C3C; border: 1px solid #555; border-radius: 5px; min-height: 40px;")
        layout.addWidget(self.status_label)

        users_group = QGroupBox("Gestión de Usuarios")
        users_layout = QVBoxLayout(users_group)
        self.user_list_widget = QListWidget()
        self.user_list_widget.setStyleSheet("QListWidget::item { padding: 5px; border-bottom: 1px solid #3C3C3C; }")
        users_layout.addWidget(self.user_list_widget, 1)
        
        user_buttons_layout = QHBoxLayout()
        
        self.toggle_photos_button = QPushButton("Mostrar Fotos")
        self.toggle_photos_button.setCheckable(True)
        self.toggle_photos_button.setChecked(self.photos_are_visible)
        self.toggle_photos_button.setToolTip("Muestra u oculta las fotos de los usuarios.\nRequiere clave si el modo privacidad está activo.")
        self.toggle_photos_button.clicked.connect(self._toggle_photo_visibility)
        
        self.settings_button = QPushButton("Configuración")
        self.settings_button.setToolTip("Abre la ventana de configuración del sistema (requiere clave).")
        self.settings_button.clicked.connect(self._open_settings_dialog)
        
        user_buttons_layout.addWidget(self.toggle_photos_button)
        user_buttons_layout.addWidget(self.settings_button)
        users_layout.addLayout(user_buttons_layout)
        
        self.toggle_video_button = QPushButton("Ocultar Vídeo")
        self.toggle_video_button.setCheckable(True)
        self.toggle_video_button.setToolTip("Oculta el stream de vídeo para ahorrar recursos.\nLa detección seguirá funcionando en segundo plano.")
        self.toggle_video_button.toggled.connect(self._toggle_video_visibility_action)
        users_layout.addWidget(self.toggle_video_button)
        
        layout.addWidget(users_group, 1)
        
        return panel

    def _create_right_panel(self) -> QTabWidget:
        """Crea el panel derecho completo, que es un sistema de pestañas."""
        self.tabs = QTabWidget()
        
        monitoring_tab = QWidget()
        monitoring_layout = QVBoxLayout(monitoring_tab)
        
        self.camera_label = QLabel("Iniciando Cámara...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 5px;")
        monitoring_layout.addWidget(self.camera_label, 3)

        fatigue_panel = self._create_fatigue_panel()
        monitoring_layout.addWidget(fatigue_panel, 1)
        self.tabs.addTab(monitoring_tab, "Monitoreo en Vivo")

        analytics_tab = self._create_analytics_tab()
        self.tabs.addTab(analytics_tab, "Analíticas y Reportes")
        
        return self.tabs
        
    def _create_fatigue_panel(self) -> QFrame:
        """Crea el panel inferior de la pestaña de monitoreo."""
        panel = QFrame()
        panel.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 5, 0, 0)
        
        metrics_group = QGroupBox("Métricas en Tiempo Real")
        metrics_layout = QFormLayout(metrics_group)
        self.ear_label = QLabel("N/A"); self.mar_label = QLabel("N/A")
        self.puc_label = QLabel("N/A"); self.moe_label = QLabel("N/A")
        metrics_layout.addRow("EAR (Ojos):", self.ear_label)
        metrics_layout.addRow("MAR (Boca):", self.mar_label)
        metrics_layout.addRow("PUC (Pupila):", self.puc_label)
        metrics_layout.addRow("MOE (Ratio):", self.moe_label)
        
        log_group = QGroupBox("Registro de Eventos de Sesión")
        log_layout = QVBoxLayout(log_group)
        self.fatigue_event_tree = QTreeWidget()
        self.fatigue_event_tree.setColumnCount(3)
        self.fatigue_event_tree.setHeaderLabels(["Evento", "Habilitado", "Cantidad"])
        self.fatigue_event_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.fatigue_event_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.fatigue_event_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.fatigue_event_tree.itemDoubleClicked.connect(self._toggle_event_status)
        log_layout.addWidget(self.fatigue_event_tree)
        self._populate_event_tree()

        layout.addWidget(metrics_group, 1)
        layout.addWidget(log_group, 2)
        return panel
    
    def _create_analytics_tab(self) -> QWidget:
        """Crea el widget completo para la pestaña de Analíticas."""
        analytics_widget = QWidget()
        main_layout = QVBoxLayout(analytics_widget)

        filters_group = QGroupBox("Filtros del Reporte")
        filters_layout = QHBoxLayout(filters_group)

        filters_layout.addWidget(QLabel("Usuario:"))
        self.analytics_user_combo = QComboBox()
        self.analytics_user_combo.setMinimumWidth(150)
        self.analytics_user_combo.setToolTip("Seleccione un usuario para ver su historial o 'Todos' para ver el historial completo.")
        filters_layout.addWidget(self.analytics_user_combo)

        filters_layout.addWidget(QLabel("Desde:"))
        self.analytics_start_date = QDateEdit(QDate.currentDate().addMonths(-1))
        self.analytics_start_date.setCalendarPopup(True)
        self.analytics_start_date.setDisplayFormat("yyyy-MM-dd")
        filters_layout.addWidget(self.analytics_start_date)

        filters_layout.addWidget(QLabel("Hasta:"))
        self.analytics_end_date = QDateEdit(QDate.currentDate())
        self.analytics_end_date.setCalendarPopup(True)
        self.analytics_end_date.setDisplayFormat("yyyy-MM-dd")
        filters_layout.addWidget(self.analytics_end_date)
        
        filters_layout.addStretch()

        self.generate_report_button = QPushButton("Generar Reporte")
        self.generate_report_button.setToolTip("Genera y muestra el reporte de eventos con los filtros seleccionados.")
        self.generate_report_button.clicked.connect(self._handle_generate_report)
        filters_layout.addWidget(self.generate_report_button)
        main_layout.addWidget(filters_group)

        kpi_group = QGroupBox("Resumen del Reporte (KPIs)")
        kpi_layout = QFormLayout(kpi_group)
        self.kpi_total_events_label = QLabel("N/A")
        self.kpi_most_frequent_label = QLabel("N/A")
        self.kpi_peak_hour_label = QLabel("N/A")
        kpi_layout.addRow("Total de Eventos:", self.kpi_total_events_label)
        kpi_layout.addRow("Evento Más Frecuente:", self.kpi_most_frequent_label)
        kpi_layout.addRow("Hora Pico de Fatiga:", self.kpi_peak_hour_label)
        main_layout.addWidget(kpi_group)

        log_group = QGroupBox("Historial de Eventos Detallado")
        log_layout = QVBoxLayout(log_group)
        self.report_table = QTableWidget()
        self.report_table.setColumnCount(4)
        self.report_table.setHorizontalHeaderLabels(["Fecha y Hora", "Usuario", "Tipo de Evento", "Detalles"])
        self.report_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.report_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.report_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.report_table.setSelectionBehavior(QTableWidget.SelectRows)
        log_layout.addWidget(self.report_table)
        main_layout.addWidget(log_group, 1)

        return analytics_widget

    # gui.py (Parte 3 de 6 - Añadir estos métodos a la clase MainWindow)

    # --- MÉTODOS DE CARGA DE DATOS Y CONFIGURACIÓN INICIAL ---

    def load_users_to_list(self):
        """
        Carga la lista de usuarios desde la base de datos y la puebla en el QListWidget.
        Maneja el modo de privacidad para mostrar fotos o íconos genéricos.
        También puebla el ComboBox de la pestaña de analíticas.
        """
        self.user_list_widget.clear()
        self.analytics_user_combo.clear()
        self.analytics_user_combo.addItem("Todos los Usuarios", 0) # userData = 0 para todos

        try:
            users = self.db_manager.get_user_details_for_list()
            if not users:
                no_user_item = QListWidgetItem("No hay usuarios registrados.")
                no_user_item.setTextAlignment(Qt.AlignCenter)
                no_user_item.setFlags(no_user_item.flags() & ~Qt.ItemIsSelectable)
                self.user_list_widget.addItem(no_user_item)
                return

            for user in users:
                user_id = user['id']
                user_code = user['codigo_usuario']
                image_path = user.get('ruta_imagen')

                # --- Poblar la lista de la GUI principal ---
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                item_layout.setContentsMargins(5, 5, 5, 5)
                item_layout.setSpacing(10)

                image_label = UserImageLabel(user_id, user_code, image_path or "", self)
                image_label.setFixedSize(QSize(64, 64))
                image_label.setStyleSheet("border: 1px solid #555; border-radius: 5px; background-color: #3C3C3C;")
                image_label.setAlignment(Qt.AlignCenter)
                
                image_label.viewUserDetails.connect(self.view_user_details)
                image_label.deleteUser.connect(self.delete_user)

                if self.photos_are_visible and image_path and os.path.exists(image_path):
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        image_label.setPixmap(pixmap.scaled(QSize(64, 64), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        image_label.setText("Error\nImg")
                        image_label.setStyleSheet(image_label.styleSheet() + "color: red;")
                else:
                    placeholder_icon = self.style().standardIcon(QStyle.SP_UserIcon)
                    image_label.setPixmap(placeholder_icon.pixmap(QSize(50, 50)))

                text_label = QLabel(f"<b>{user_code}</b><br/><small>(ID: {user_id})</small>")
                text_label.setWordWrap(True)

                item_layout.addWidget(image_label)
                item_layout.addWidget(text_label, 1)

                list_item = QListWidgetItem(self.user_list_widget)
                list_item.setSizeHint(item_widget.sizeHint())
                self.user_list_widget.addItem(list_item)
                self.user_list_widget.setItemWidget(list_item, item_widget)
                
                # --- Poblar el ComboBox de la pestaña de analíticas ---
                self.analytics_user_combo.addItem(user_code, user_id)

        except Exception as e:
            logger.error(f"Error al cargar la lista de usuarios: {e}", exc_info=True)
            self.user_list_widget.addItem("Error al cargar usuarios.")

    def _populate_event_tree(self):
        """Puebla el árbol de eventos de fatiga con los eventos configurados y su estado."""
        self.fatigue_event_tree.clear()
        event_toggles = self.config.get('event_toggles', {})
        for event_name in sorted(event_toggles.keys()):
            is_enabled = event_toggles.get(event_name, False)
            
            item = QTreeWidgetItem(self.fatigue_event_tree)
            item.setText(0, event_name)
            item.setText(1, "✓" if is_enabled else "✗")
            item.setForeground(1, Qt.green if is_enabled else Qt.red)
            item.setToolTip(1, "Haga doble clic para cambiar el estado de detección de este evento.")
            
            item.setText(2, str(self.fatigue_event_counts.get(event_name, 0)))
            item.setTextAlignment(2, Qt.AlignCenter)
    
    def trigger_first_run_setup(self):
        """
        Inicia el diálogo de configuración de clave por primera vez.
        Este método es llamado explícitamente por main.py.
        """
        QMessageBox.information(self, "Configuración Inicial Requerida", 
                                "Bienvenido a Copiloto-ID.\n\nComo es la primera ejecución, "
                                "debe establecer una clave de administrador para proteger la configuración y la privacidad de los usuarios.")
        self._handle_change_password(is_first_run=True)

    def _handle_change_password(self, is_first_run=False):
        """Maneja el flujo completo para cambiar la clave de administrador."""
        if not is_first_run:
            current_pass, ok = QInputDialog.getText(self, "Verificación de Seguridad", "Introduzca su clave actual:", QLineEdit.Password)
            if not ok: return # El usuario canceló
            if not ConfigManager.verify_password(current_pass):
                QMessageBox.warning(self, "Acceso Denegado", "La clave actual es incorrecta.")
                return
        
        while True:
            new_pass, ok = QInputDialog.getText(self, "Crear Nueva Clave", "Introduzca la nueva clave (mín. 4 caracteres):", QLineEdit.Password)
            if not ok: return
            if new_pass and len(new_pass) >= 4:
                break
            QMessageBox.warning(self, "Clave Inválida", "La nueva clave debe tener al menos 4 caracteres.")

        confirm_pass, ok = QInputDialog.getText(self, "Confirmar Nueva Clave", "Confirme la nueva clave:", QLineEdit.Password)
        if not ok: return
        if new_pass != confirm_pass:
            QMessageBox.warning(self, "Error", "Las nuevas claves no coinciden.")
            return
            
        recovery_code = ConfigManager.set_new_password(new_pass)
        self.config = ConfigManager.load_full_config() # Recargar la config con la nueva clave
        
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Clave Cambiada Exitosamente")
        msg_box.setText(
            "La clave de administrador ha sido establecida.\n\n"
            "**¡IMPORTANTE! GUARDE ESTE CÓDIGO DE RECUPERACIÓN EN UN LUGAR SEGURO.**\n"
            "Es la única forma de recuperar el acceso si olvida su nueva clave."
        )
        msg_box.setInformativeText(f"<b>{recovery_code}</b>")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        logger.info("La clave de administrador ha sido cambiada/establecida.")    

    # gui.py (Parte 4 de 6 - Añadir estos métodos a la clase MainWindow)

    def _handle_listening_state(self, frame: np.ndarray, face_results: Any, pose_data: Optional[Tuple]):
        """Lógica para el estado de escucha, esperando un rostro estable."""
        self.status_label.setText("Estado: Buscando Usuario...")
        
        # Si se detecta un rostro Y la pose es aceptable
        if face_results and face_results.multi_face_landmarks and pose_data and PoseEstimator.is_pose_acceptable(*pose_data, self.config):
            self.state_timers['stable_face_counter'] += 1
            stable_frames_threshold = self.config['state_machine_thresholds']['stable_face_frames_to_identify']
            
            if self.state_timers['stable_face_counter'] >= stable_frames_threshold:
                logger.info("Rostro estable detectado. Pasando a estado de identificación.")
                self.app_state = "IDENTIFYING"
                self.state_timers['stable_face_counter'] = 0  # Resetear contador
        else:
            self.state_timers['stable_face_counter'] = 0  # Resetear si el rostro no es estable
            
    def _handle_identifying_state(self, frame: np.ndarray, face_results: Any):
        """
        Lógica para el estado de identificación. Se ejecuta una sola vez por intento
        para determinar si el rostro detectado pertenece a un usuario registrado.
        """
        self.status_label.setText("Estado: Identificando Usuario...")

        if not self.last_known_face_results or not self.last_known_face_results.multi_face_landmarks:
            logger.warning("Se intentó identificar pero se perdieron los landmarks faciales. Volviendo a LISTENING.")
            self.app_state = "LISTENING"
            return

        try:
            face_lm_obj = self.last_known_face_results.multi_face_landmarks[0]
            pose_data = PoseEstimator.estimate_head_pose(frame.shape[:2], face_lm_obj)
            
            if not pose_data or not PoseEstimator.is_pose_acceptable(*pose_data, self.config):
                logger.warning("Pose inaceptable durante el intento de identificación. Volviendo a LISTENING.")
                self.app_state = "LISTENING"
                return

            result = self.face_processor.align_face(frame, face_lm_obj)
            if result is None:
                logger.warning("No se pudo alinear el rostro para la identificación.")
                self.app_state = "LISTENING"
                return
            aligned_blob, _ = result
            
            query_emb = self.face_processor.get_embedding(aligned_blob)
            if query_emb is None:
                logger.error("No se pudo generar el embedding para la identificación.")
                self.app_state = "LISTENING"
                return
                
            model_name = self.config['recognition_settings']['model_filename']
            known_embs = self.db_manager.get_all_user_embeddings(model_name)
            threshold = self.config['recognition_settings']['threshold']
            
            match, similarity = self.face_processor.find_match(query_emb, known_embs, threshold)
            
            if match:
                self._login_user(match['user_id'], match['codigo_usuario'], similarity)
            else:
                logger.info(f"Usuario no reconocido (similitud máx: {similarity:.2f}). Pasando a AUTO_REGISTERING.")
                self.db_manager.log_access_log(None, "Fallido - Desconocido", similarity)
                self.app_state = "AUTO_REGISTERING"
                self.state_timers['auto_register_start_time'] = time.time()
        except Exception as e:
            logger.error(f"Excepción durante la identificación: {e}", exc_info=True)
            self.app_state = "LISTENING"

    def _handle_auto_registering_state(self, frame: np.ndarray, face_results: Any):
        """Lógica para el registro automático de un nuevo usuario si permanece estable."""
        if self.state_timers['auto_register_start_time'] is None:
            self.state_timers['auto_register_start_time'] = time.time()
            
        wait_time = self.config['state_machine_thresholds']['auto_register_seconds']
        time_elapsed = time.time() - self.state_timers['auto_register_start_time']
        time_left = wait_time - time_elapsed

        self.status_label.setText(f"Rostro nuevo detectado. Mantenga la posición.\nRegistrando en {int(max(0, time_left))} segundos...")
        
        pose_data = None
        if face_results and face_results.multi_face_landmarks:
            pose_data = PoseEstimator.estimate_head_pose(frame.shape[:2], face_results.multi_face_landmarks[0])

        is_acceptable = bool(pose_data and PoseEstimator.is_pose_acceptable(*pose_data, self.config))
        
        if not is_acceptable:
            logger.info("Candidato a registro perdido (pose inaceptable o rostro perdido). Volviendo a LISTENING.")
            self.app_state = "LISTENING"
            self.state_timers['auto_register_start_time'] = None
            return
            
        if time_left <= 0:
            logger.info("Tiempo de espera cumplido. Registrando nuevo usuario...")
            self._execute_auto_registration(frame, face_results.multi_face_landmarks[0], pose_data)
            
    def _handle_monitoring_state(self, frame: np.ndarray, face_results: Any, hand_results: Any, pose_results: Any, pose_data: Optional[Tuple]):
        """
        Lógica principal de monitoreo de fatiga, que se ejecuta en cada frame
        mientras un usuario está siendo monitoreado.
        
        Args:
            frame: El fotograma actual en formato BGR
            face_results: Resultados de la detección facial de MediaPipe
            hand_results: Resultados de la detección de manos de MediaPipe
            pose_results: Resultados de la detección de postura de MediaPipe
            pose_data: Tupla con los ángulos de rotación de la cabeza (pitch, yaw, roll)
        """
        if not hasattr(self, 'fatigue_processor'):
            return
            
        # Verificar si hay un rostro detectado
        face_detected = face_results is not None and face_results.multi_face_landmarks
        
        # Actualizar el estado de detección de rostro para el analizador de fatiga
        self.fatigue_processor.update_face_detection_status(face_detected)
        
        # Si no hay rostro, registrar evento de distracción y salir
        if not face_detected:
            self.fatigue_processor.handle_distraction()
            return
            
        # Obtener los landmarks faciales
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Procesar el frame para detección de fatiga
        detected_events, overlay_data = self.fatigue_processor.process_frame_for_inference(
            frame, 
            face_landmarks, 
            hand_results,  # Pasar resultados de manos
            pose_results,  # Pasar resultados de postura
            pose_data
        )
        
        # Actualizar la interfaz con los resultados
        self.last_monitoring_data = overlay_data
        self.update_facial_metrics_display(overlay_data.get('metrics', {}))
        
        # Manejar eventos de fatiga detectados
        if detected_events:
            self._handle_fatigue_events(detected_events)
            
        # Actualizar el temporizador de inactividad
        self.inactivity_timer = time.time()

    def _execute_auto_registration(self, frame, face_landmarks, pose_data):
        """Contiene la lógica completa para crear un nuevo usuario y sus datos."""
        try:
            result = self.face_processor.align_face(frame, face_landmarks)
            if result is None: 
                self.app_state = "LISTENING"
                return
            aligned_blob, aligned_face_img = result

            emb = self.face_processor.get_embedding(aligned_blob)
            if emb is None: 
                self.app_state = "LISTENING"
                return
                
            new_user_id, new_user_code = self.db_manager.register_user()
            if new_user_id and new_user_code:
                os.makedirs("rostros_capturados", exist_ok=True)
                image_path = os.path.join("rostros_capturados", f"{new_user_code}_{int(time.time())}.png")
                cv2.imwrite(image_path, aligned_face_img)
                self.db_manager.add_user_embedding(new_user_id, emb, image_path, self.config['recognition_settings']['model_name'], pose_data)
                
                logger.info(f"Nuevo usuario '{new_user_code}' registrado automáticamente.")
                QMessageBox.information(self, "Registro Automático", f"Nuevo usuario '{new_user_code}' ha sido registrado en el sistema.")
                self.load_users_to_list()
                
                self._login_user(new_user_id, new_user_code, 1.0) # Similitud de 1.0 para nuevos registros
            else:
                self.app_state = "LISTENING"
        except Exception as e:
            logger.error(f"Excepción durante el registro automático: {e}", exc_info=True)
            self.app_state = "LISTENING"

    def _prepare_for_monitoring_session(self):
        """Prepara todas las variables para una nueva sesión de monitoreo."""
        logger.info(f"Preparando nueva sesión de monitoreo para {self.current_user_code}.")
        self.fatigue_processor.reset_state()
        self.fatigue_event_counts.clear()
        self._populate_event_tree()
        
        self.state_timers['enrichment_start_time'] = time.time()
        self.state_timers['session_embeddings_captured'] = 0
        self.state_timers['session_captured_poses'] = []
        self.last_monitoring_data = {}
        
        self.is_calibrating = False
        self.calibration_frames_data = []
        self.fatigue_processor.calibration_data = None

    def _login_user(self, user_id, user_code, similarity):
        """Centraliza la lógica para iniciar sesión de un usuario."""
        self.current_user_id = user_id
        self.current_user_code = user_code
        logger.info(f"Usuario autenticado: {self.current_user_code} (ID: {self.current_user_id}) con similitud {similarity:.2f}.")
        self.status_label.setText(f"Bienvenido, {self.current_user_code}")
        self.db_manager.log_access_log(self.current_user_id, "Exitoso", similarity)
        self.app_state = "MONITORING"
        self._prepare_for_monitoring_session()

    def _logout_user(self):
        """Resetea la sesión del usuario actual y vuelve al estado de escucha."""
        logger.info(f"Usuario {self.current_user_code} ausente. Finalizando sesión de monitoreo.")
        
        # Si la opción está activada en la configuración, guardar el reporte de la sesión que acaba de terminar.
        if self.config.get('reporting_settings', {}).get('SAVE_LOG_ON_SESSION_END', False):
            if self.current_user_code and self.current_user_code != "N/A":
                logger.info(f"Guardando reporte de sesión para el usuario {self.current_user_code}.")
                # Llamar a la función mejorada con el código del usuario
                notificaciones.save_event_log_to_file(user_code=self.current_user_code)

        # Resto de la lógica de cierre de sesión
        self.status_label.setText("Sesión finalizada. Buscando usuario...")
        
        self.current_user_id = None
        self.current_user_code = "N/A"
        self.app_state = "LISTENING"
        self.state_timers['face_lost_counter'] = 0
        self.fatigue_processor.reset_state()
        self.fatigue_event_counts.clear()
        self._populate_event_tree()
        
        if not is_user_present:
            self.state_timers['face_lost_counter'] += 1
            logout_threshold = self.config['state_machine_thresholds']['face_lost_frames_to_logout']
            if self.state_timers['face_lost_counter'] > logout_threshold:
                self._logout_user() # Cierra la sesión y vuelve a LISTENING
            return
        
        # Si el usuario está presente, se resetea el contador de ausencia.
        self.state_timers['face_lost_counter'] = 0
        
        # --- 2. Lógica de Calibración Inteligente ---
        if self.fatigue_processor.calibration_data is None and not self.is_calibrating:
            logger.info(f"Verificando calibración para el usuario ID: {self.current_user_id}")
            cal_data = self.db_manager.load_calibration_data(self.current_user_id)
            if cal_data:
                self.fatigue_processor.set_calibration(cal_data)
                QMessageBox.information(self, "Calibración", f"Calibración para {self.current_user_code} cargada desde la base de datos.")
            else:
                logger.info(f"No se encontraron datos de calibración. Iniciando calibración en tiempo real.")
                self.is_calibrating = True
                self.calibration_frames_data = []
                self.status_label.setText(f"CALIBRANDO para {self.current_user_code}...")
        
        if self.is_calibrating:
            cal_metrics = self.fatigue_processor.process_frame_for_calibration(face_results, frame.shape[:2])
            if cal_metrics:
                self.calibration_frames_data.append(cal_metrics)
            
            # Recopilar datos durante aprox. 5 segundos (asumiendo ~20fps)
            if len(self.calibration_frames_data) >= 100:
                df = pd.DataFrame(self.calibration_frames_data)
                final_cal_data = {
                    'cal_ear_mean': df['ear'].mean(), 'cal_ear_std': df['ear'].std(),
                    'cal_mar_mean': df['mar'].mean(), 'cal_mar_std': df['mar'].std(),
                    'cal_puc_mean': df['puc'].mean(), 'cal_puc_std': df['puc'].std(),
                    'cal_moe_mean': df['moe'].mean(), 'cal_moe_std': df['moe'].std(),
                }
                self.fatigue_processor.set_calibration(final_cal_data)
                self.db_manager.save_calibration_data(self.current_user_id, final_cal_data)
                self.is_calibrating = False
                logger.info(f"Calibración en tiempo real completada y guardada para {self.current_user_code}.")
                QMessageBox.information(self, "Calibración", "Calibración completada exitosamente.")
            
            # No continuar con el resto del procesamiento mientras se calibra
            return

        # --- 3. Lógica de Enriquecimiento de Perfil ---
        enrich_config = self.config.get('profile_enrichment', {})
        if enrich_config.get('enabled') and self.state_timers.get('enrichment_start_time'):
            if (time.time() - self.state_timers['enrichment_start_time']) < 60: # 1 minuto
                if self.state_timers['session_embeddings_captured'] < enrich_config.get('embeddings_per_session_target', 5):
                    
                    # Comprobar si la pose es suficientemente nueva
                    is_new_pose = True
                    if self.state_timers['session_captured_poses']:
                        current_pose_np = np.array(pose_data)
                        min_dist = min([np.linalg.norm(current_pose_np - p) for p in self.state_timers['session_captured_poses']])
                        if min_dist < enrich_config.get('min_pose_difference_threshold', 15.0):
                            is_new_pose = False
                    
                    if is_new_pose:
                        logger.info("Detectada nueva pose para enriquecimiento de perfil.")
                        result = self.face_processor.align_face(frame, face_results.multi_face_landmarks[0])
                        if result:
                            aligned_blob, aligned_face_img = result
                            emb = self.face_processor.get_embedding(aligned_blob)
                            if emb is not None:
                                image_path = os.path.join("rostros_capturados", f"{self.current_user_code}_extra_{int(time.time())}.png")
                                cv2.imwrite(image_path, aligned_face_img)
                                self.db_manager.add_user_embedding(self.current_user_id, emb, image_path, self.config['recognition_settings']['model_filename'], pose_data)
                                
                                self.state_timers['session_captured_poses'].append(np.array(pose_data))
                                self.state_timers['session_embeddings_captured'] += 1
                                logger.info(f"Embedding de enriquecimiento #{self.state_timers['session_embeddings_captured']} guardado para {self.current_user_code}.")
                                self.status_label.setText("Perfil facial mejorado...")
            else:
                # Desactivar el enriquecimiento para esta sesión una vez pasado el minuto
                self.state_timers['enrichment_start_time'] = None
                logger.info("Período de enriquecimiento de perfil finalizado para esta sesión.")

        # --- 4. Detección Principal de Fatiga ---
        hand_results = None # Se obtendría de un procesador de manos de MediaPipe si estuviera activo
        pose_results_full = None # Se obtendría del procesador de pose de MediaPipe
        
        events, overlay_data = self.fatigue_processor.process_frame_for_inference(
            frame, face_results, hand_results, pose_results_full
        )
        self.last_monitoring_data = overlay_data
        self.last_monitoring_data['pose_data'] = pose_data # Añadir datos de pose para el overlay

        # --- 5. Registro y Notificación de Eventos ---
        if events:
            # Eliminar eventos que no están activados en la configuración
            active_events = [e for e in events if self.config.get('event_toggles', {}).get(e, False)]
            
            if active_events:
                self.update_fatigue_event_log(active_events)
                for event in active_events:
                    notificaciones.speak(event)
                    self.db_manager.log_behavior_event(self.current_user_id, event)
                    
        # Actualizar las métricas en el panel lateral de la GUI
        if 'metrics' in overlay_data:
            self.update_facial_metrics_display(overlay_data['metrics'])

    # gui.py (Parte 6 de 6 - Añadir estos métodos a la clase MainWindow para completarla)

    # --- LÓGICA DE ACTUALIZACIÓN DE LA INTERFAZ GRÁFICA ---

    def _update_display(self, frame: np.ndarray, face_results: Any):
        """
        Actualiza el widget de video. Si el video es visible, aplica los overlays
        y lo muestra en el QLabel principal.
        """
        if not self.video_is_visible:
            return

        pose_data = None
        if face_results and face_results.multi_face_landmarks:
            pose_data = PoseEstimator.estimate_head_pose(frame.shape[:2], face_results.multi_face_landmarks[0])

        # Dibuja todas las capas de información sobre el fotograma
        final_frame = self._draw_frame_overlays(frame, face_results, None, None, pose_data, self.last_monitoring_data)
        
        # Convierte el fotograma de formato OpenCV (BGR) a formato Qt (RGB) y lo muestra
        try:
            h, w, ch = final_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(final_frame.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"Error al convertir y mostrar el fotograma: {e}")

    def _draw_frame_overlays(self, frame: np.ndarray, face_results: Any, hand_results: Any, pose_results: Any, pose_data: Optional[Tuple], overlay_data: Dict) -> np.ndarray:
        """
        Dibuja toda la información visual sobre un fotograma.
        
        Args:
            frame: El fotograma de entrada en formato BGR
            face_results: Resultados de la detección facial de MediaPipe
            hand_results: Resultados de la detección de manos de MediaPipe
            pose_results: Resultados de la detección de postura de MediaPipe
            pose_data: Tupla con los ángulos de rotación de la cabeza (pitch, yaw, roll)
            overlay_data: Diccionario con datos adicionales para mostrar
            
        Returns:
            El fotograma con los overlays dibujados
        """
        # Dibujar malla facial si hay un rostro detectado
        if self.app_state != "REPOSO" and face_results and face_results.multi_face_landmarks:
            self.face_processor.draw_face_mesh(frame, face_results.multi_face_landmarks[0])

        # Opcional: Dibujar landmarks de manos y pose (para depuración o visualización)
        if self.app_state != "REPOSO" and hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.face_processor.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    self.face_processor.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.face_processor.mp_drawing_styles.get_default_hand_connections_style()
                )
                
        if self.app_state != "REPOSO" and pose_results and pose_results.pose_landmarks:
            self.face_processor.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                self.face_processor.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Dibujar estado actual de la aplicación
        status_text = f"ESTADO: {self.app_state}"
        if self.app_state == "MONITORING" and self.current_user_code:
            status_text += f" ({self.current_user_code})"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Dibujar métricas de fatiga en tiempo real
        metrics = overlay_data.get('metrics', {})
        if metrics:
            y_pos = 60
            for key, value in metrics.items():
                if key != 'landmarks_np':
                    cv2.putText(frame, f"{key.upper()}: {value:.2f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_pos += 20
        
        # Dibujar pose
        if pose_data:
            pitch, yaw, roll = pose_data
            pose_text = f"P: {pitch:.1f} Y: {yaw:.1f} R: {roll:.1f}"
            cv2.putText(frame, pose_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Dibujar alerta de texto principal
        alert_text = overlay_data.get('alert_text')
        if alert_text:
            font_scale = 1.5
            thickness = 3
            (text_w, text_h), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)
            text_x = int((frame.shape[1] - text_w) / 2)
            text_y = int(frame.shape[0] * 0.8)
            # Fondo negro para mejor legibilidad
            cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), thickness + 3)
            # Texto en rojo
            cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 255), thickness)

        # Mostrar texto de pausa activa si está próximo
        pausa_text = overlay_data.get('pausa_text')
        if pausa_text:
            cv2.putText(frame, pausa_text, (frame.shape[1] - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Dibujar recuadro rojo de alerta máxima
        if overlay_data.get('is_max_alert'):
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 20)
            
        return frame

    def update_facial_metrics_display(self, metrics: Dict):
        """Actualiza las etiquetas del panel de métricas."""
        self.ear_label.setText(f"{metrics.get('ear', 0):.3f}")
        self.mar_label.setText(f"{metrics.get('mar', 0):.3f}")
        self.puc_label.setText(f"{metrics.get('puc', 0):.3f}")
        self.moe_label.setText(f"{metrics.get('moe', 0):.3f}")

    def update_fatigue_event_log(self, events: List[str]):
        """Actualiza el árbol de eventos de fatiga con nuevos eventos detectados."""
        for event in events:
            self.fatigue_event_counts[event] += 1
        self._populate_event_tree()

    # --- SLOTS Y MANEJADORES DE EVENTOS DE USUARIO ---

    def _open_settings_dialog(self):
        """Abre el diálogo de configuración con protección por clave."""
        password, ok = QInputDialog.getText(self, "Acceso Restringido", "Introduzca la clave de administrador:", QLineEdit.Password)
        if ok and password:
            if ConfigManager.verify_password(password):
                dialog = SettingsDialog(copy.deepcopy(self.config), self)
                dialog.settings_saved.connect(self._on_settings_saved)
                dialog.exec_()
            else:
                QMessageBox.warning(self, "Acceso Denegado", "La clave es incorrecta.")

    def _on_settings_saved(self):
        """Recarga la configuración después de que se haya guardado."""
        logger.info("Recargando configuración en la ventana principal tras guardado.")
        self.config = ConfigManager.load_full_config()
        self.fatigue_processor = FatigueProcessor(self.config)
        self._populate_event_tree()
        QMessageBox.information(self, "Configuración", "Los ajustes se han guardado.\nAlgunos cambios pueden requerir un reinicio de la sesión de monitoreo.")

    def _toggle_photo_visibility(self, checked):
        """Maneja el clic en el botón de mostrar/ocultar fotos."""
        if checked:
            password, ok = QInputDialog.getText(self, "Acceso Restringido", "Introduzca la clave para mostrar las fotos:", QLineEdit.Password)
            if ok and password and ConfigManager.verify_password(password):
                self.photos_are_visible = True
            else:
                if ok: QMessageBox.warning(self, "Acceso Denegado", "La clave es incorrecta.")
                self.toggle_photos_button.setChecked(False)
                return
        else:
            self.photos_are_visible = False
            
        self.toggle_photos_button.setText("Ocultar Fotos" if self.photos_are_visible else "Mostrar Fotos")
        self.load_users_to_list()

    def _toggle_video_visibility_action(self, checked):
        """Maneja el botón de ocultar/mostrar video."""
        self.video_is_visible = not checked
        if checked:
            self.toggle_video_button.setText("Mostrar Vídeo")
            self.camera_label.setText("MONITOREO EN SEGUNDO PLANO...")
            self.camera_label.setStyleSheet("background-color: #111; color: #0f0; font-family: 'Courier New', monospace; font-size: 18pt; border-radius: 5px;")
            logger.info("Video ocultado. La detección continúa en segundo plano.")
        else:
            self.toggle_video_button.setText("Ocultar Vídeo")
            self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 5px;")
            logger.info("Visualización de video reactivada.")

    def _toggle_event_status(self, item: QTreeWidgetItem, column: int):
        """Manejador para activar/desactivar un evento desde el árbol de la GUI."""
        if column != 1: return
        
        event_name = item.text(0)
        if event_name in self.config['event_toggles']:
            current_status = self.config['event_toggles'][event_name]
            new_status = not current_status
            self.config['event_toggles'][event_name] = new_status
            ConfigManager.save_full_config(self.config)
            
            item.setText(1, "✓" if new_status else "✗")
            item.setForeground(1, Qt.green if new_status else Qt.red)
            logger.info(f"El evento '{event_name}' ha sido {'habilitado' if new_status else 'deshabilitado'}.")

    def _handle_generate_report(self):
        """Inicia la generación de un reporte de analíticas en un hilo secundario."""
        user_id = self.analytics_user_combo.currentData()
        start_date = self.analytics_start_date.date().toString("yyyy-MM-dd")
        end_date = self.analytics_end_date.date().toString("yyyy-MM-dd")

        self.generate_report_button.setEnabled(False)
        self.generate_report_button.setText("Generando...")

        self.analytics_worker = AnalyticsWorker(self.db_manager, user_id, start_date, end_date)
        self.analytics_worker_thread = QThread()
        self.analytics_worker.moveToThread(self.analytics_worker_thread)
        self.analytics_worker.finished.connect(self._on_report_finished)
        self.analytics_worker.error.connect(self._on_report_error)
        self.analytics_worker_thread.started.connect(self.analytics_worker.run)
        self.analytics_worker_thread.finished.connect(self.analytics_worker_thread.deleteLater)
        self.analytics_worker_thread.start()

    def _on_report_finished(self, result):
        """Slot que se activa cuando el worker de analíticas termina."""
        self.generate_report_button.setEnabled(True)
        self.generate_report_button.setText("Generar Reporte")
        
        if isinstance(result, Exception):
            self._on_report_error(result)
            return

        kpis = result['kpis']
        events_data = result['events_data']
        
        self.kpi_total_events_label.setText(str(kpis.get('total_events', 0)))
        mf_event, mf_count = kpis.get('most_frequent_event', ("N/A", 0))
        self.kpi_most_frequent_label.setText(f"{mf_event} ({mf_count} veces)")
        ph_hour, ph_count = kpis.get('peak_hour', ("N/A", 0))
        self.kpi_peak_hour_label.setText(f"{ph_hour:02d}:00 - {ph_hour+1:02d}:00 ({ph_count} eventos)")

        self.report_table.setRowCount(len(events_data))
        for row_idx, event in enumerate(events_data):
            self.report_table.setItem(row_idx, 0, QTableWidgetItem(event.get('fecha_hora', '')))
            self.report_table.setItem(row_idx, 1, QTableWidgetItem(event.get('codigo_usuario', 'N/A')))
            self.report_table.setItem(row_idx, 2, QTableWidgetItem(event.get('tipo_evento', '')))
            self.report_table.setItem(row_idx, 3, QTableWidgetItem(str(event.get('duracion_seg', '')) if event.get('duracion_seg') else ''))
        
        logger.info(f"Reporte generado y mostrado con {len(events_data)} eventos.")

    def _on_report_error(self, error: Exception):
        """Slot que se activa si el worker de analíticas falla."""
        QMessageBox.critical(self, "Error de Reporte", f"No se pudo generar el reporte:\n{str(error)}")
        self.generate_report_button.setEnabled(True)
        self.generate_report_button.setText("Generar Reporte")

    # --- GESTIÓN DE HILOS Y CICLO DE VIDA ---
    
    def _update_camera_selector(self):
        """Actualiza el ComboBox con la lista de cámaras disponibles."""
        current_cam = self.camera_combo.currentData() if self.camera_combo.count() > 0 else None
        
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        
        if not self.available_cameras:
            self.camera_combo.addItem("No hay cámaras disponibles", -1)
            self.camera_combo.setEnabled(False)
            self.camera_status.setText("Estado: Sin cámaras detectadas")
            self.camera_status.setStyleSheet("color: red;")
        else:
            for idx in self.available_cameras:
                self.camera_combo.addItem(f"Cámara {idx}", idx)
            
            # Seleccionar la cámara actual si está disponible
            if self.camera_index in self.available_cameras:
                index = self.available_cameras.index(self.camera_index)
                self.camera_combo.setCurrentIndex(index)
            else:
                # Si la cámara actual no está disponible, seleccionar la primera
                self.camera_index = self.available_cameras[0]
                self.camera_combo.setCurrentIndex(0)
                
            self.camera_combo.setEnabled(True)
            self.camera_status.setText("Estado: Lista")
            self.camera_status.setStyleSheet("color: green;")
        
        self.camera_combo.blockSignals(False)
    
    def _on_camera_changed(self, index):
        """Maneja el cambio de selección de cámara."""
        if index < 0:
            return
            
        new_camera = self.camera_combo.currentData()
        if new_camera == -1 or new_camera == self.camera_index:
            return
            
        logger.info(f"Solicitado cambio a cámara {new_camera} (actual: {self.camera_index})")
        was_capturing = self.capturing
        
        if was_capturing:
            self.stop_camera()
            QApplication.processEvents()
            time.sleep(0.5)
        
        self.camera_index = new_camera
        self.config['camera_settings']['camera_index'] = self.camera_index
        ConfigManager.save_full_config(self.config)
        
        if was_capturing:
            self.start_camera()
    
    def _refresh_camera_list(self):
        """Actualiza la lista de cámaras disponibles."""
        logger.info("Actualizando lista de cámaras...")
        was_capturing = self.capturing
        
        if was_capturing:
            self.stop_camera()
            QApplication.processEvents()
            time.sleep(0.5)
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            max_idx_to_check = self.config['camera_settings'].get('MAX_CAMERA_INDEX_TO_CHECK', 4)
            self.available_cameras = self._detect_available_cameras(max_idx_to_check)
            
            # Actualizar el índice de cámara actual si es necesario
            if self.available_cameras:
                if self.camera_index not in self.available_cameras:
                    self.camera_index = self.available_cameras[0]
                    self.config['camera_settings']['camera_index'] = self.camera_index
                    ConfigManager.save_full_config(self.config)
            else:
                self.camera_index = -1
            
            self._update_camera_selector()
            
            if self.available_cameras:
                self.status_label.setText("Lista de cámaras actualizada correctamente.")
                if was_capturing:
                    self.start_camera()
            else:
                self.status_label.setText("No se detectaron cámaras. Conecte una y actualice.")
                
        except Exception as e:
            error_msg = f"Error al actualizar la lista de cámaras: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: red;")
        finally:
            QApplication.restoreOverrideCursor()
    
    def start_camera(self):
        """Inicia el hilo de la cámara si no está corriendo."""
        if self.capturing:
            logger.warning("Intento de iniciar cámara que ya está activa.")
            return
            
        if self.camera_index == -1:
            QMessageBox.warning(self, "Sin Cámara", "No hay una cámara seleccionada o disponible.")
            self.camera_status.setText("Estado: Sin cámara seleccionada")
            self.camera_status.setStyleSheet("color: red;")
            return
            
        try:
            # Si hay un hilo de cámara anterior, asegurarse de limpiarlo
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
                self.camera_thread.wait(1000)
                self.camera_thread = None
                
            # Crear un nuevo hilo de cámara con la resolución configurada
            frame_width = self.config['camera_settings'].get('FRAME_WIDTH', 1280)
            frame_height = self.config['camera_settings'].get('FRAME_HEIGHT', 720)
            
            self.camera_thread = CameraThread(
                self.camera_index, 
                self, 
                frame_width=frame_width,
                frame_height=frame_height
            )
            self.camera_thread.update_frame.connect(self.process_frame)
            self.camera_thread.error.connect(self._on_camera_error)
            
            self.camera_thread.start()
            self.capturing = True
            self.camera_status.setText(f"Estado: Conectando a cámara {self.camera_index}...")
            self.camera_status.setStyleSheet("color: orange;")
            
            # Deshabilitar controles mientras la cámara está activa
            self.camera_combo.setEnabled(False)
            self.refresh_button.setEnabled(False)
            
        except Exception as e:
            error_msg = f"Error al iniciar la cámara: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Error de Cámara", error_msg)
            self.camera_status.setText("Estado: Error de conexión")
            self.camera_status.setStyleSheet("color: red;")
            self.capturing = False
            
    def _on_camera_error(self, error_msg):
        """Maneja errores reportados por el hilo de la cámara."""
        logger.error(f"Error de cámara: {error_msg}")
        QMessageBox.critical(self, "Error de Cámara", error_msg)
        self.stop_camera()

    def stop_camera(self):
        """Detiene el hilo de la cámara de forma segura."""
        if not self.capturing:
            return
            
        logger.info("Deteniendo cámara...")
        
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait(1000)
            self.camera_thread = None
            
        self.capturing = False
        
        # Actualizar interfaz
        if not self.available_cameras or self.camera_index == -1:
            self.camera_status.setText("Estado: Sin cámaras disponibles")
            self.camera_status.setStyleSheet("color: red;")
        else:
            self.camera_status.setText(f"Estado: Cámara {self.camera_index} lista")
            self.camera_status.setStyleSheet("color: green;")
            
        # Re-habilitar controles
        self.camera_combo.setEnabled(bool(self.available_cameras))
        self.refresh_button.setEnabled(True)
            
    def _start_mpu_thread(self):
        self.mpu_thread = MPUThread(self.config, self)
        self.mpu_thread.motion_detected.connect(self._wake_from_sleep)
        self.mpu_thread.no_motion_detected_for_duration.connect(self._enter_sleep_mode)
        self.mpu_thread.start()
        
    def _enter_sleep_mode(self):
        if self.app_state == "REPOSO": return
        logger.warning("Inactividad prolongada detectada. Entrando en modo reposo.")
        self.app_state = "REPOSO"
        self.stop_camera()
        self.camera_label.setText("MODO REPOSO\n(Esperando movimiento)")
        self.camera_label.setStyleSheet("background-color: #111; color: #fff; font-family: 'Courier New', monospace; font-size: 22pt; border-radius: 5px;")

    def _wake_from_sleep(self):
        if self.app_state != "REPOSO": return
        logger.info("Movimiento detectado. Saliendo del modo reposo.")
        self.app_state = "LISTENING"
        self.camera_label.setText("Reconectando cámara...")
        self.camera_label.setStyleSheet("background-color: #000; border: 1px solid #444; border-radius: 5px;")
        self.start_camera()

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Procesa cada fotograma capturado por la cámara, realiza las detecciones
        de visión artificial y actualiza el estado de la aplicación y la GUI.
        """
        if not hasattr(self, 'face_processor') or not self.face_processor.initialized:
            # Si el procesador facial no está listo, solo muestra el frame si es visible
            if self.video_is_visible:
                try:
                    h, w, ch = frame_bgr.shape
                    bytes_per_line = ch * w
                    q_image = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
                    self.camera_label.setPixmap(QPixmap.fromImage(q_image))
                except Exception as e:
                    logger.error(f"Error al mostrar frame sin procesador facial listo: {e}")
            return frame_bgr
        
        # 1. Asegurar ROI y ajuste de exposición (se aplica al frame original)
        if hasattr(roi_autoexp, 'auto_center_enabled') and roi_autoexp.auto_center_enabled:
            height, width = frame_bgr.shape[:2]
            roi_autoexp.ensure_roi_centered(width, height, self.config.get('roi_autoexposure_settings', {}).get('CENTER_CONFIG'))
        
        # Si no hay ROI, inicializar uno centrado
        if not roi_autoexp.roi_selected:
            height, width = frame_bgr.shape[:2]
            roi_autoexp.init_centered_roi(width, height, self.config.get('roi_autoexposure_settings', {}).get('CENTER_CONFIG'))
        
        # Ajustar exposición usando el objeto CAP de la cámara
        if 'roi_autoexposure_settings' in self.config and self.camera_thread and self.camera_thread.cap:
            frame_bgr = roi_autoexp.adjust_exposure(frame_bgr, self.camera_thread.cap, self.config)
        
        # 2. Realizar detecciones de MediaPipe
        # Convertir a RGB para MediaPipe (MediaPipe prefiere RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Optimización de rendimiento para MediaPipe

        # Realizar todas las detecciones de MediaPipe
        face_results = self.face_processor.face_mesh.process(frame_rgb)
        hand_results = self.face_processor.hands.process(frame_rgb)  # Procesar manos
        pose_results = self.face_processor.pose.process(frame_rgb)    # Procesar pose de cuerpo

        frame_rgb.flags.writeable = True  # Volver a habilitar escritura si se modifica en el futuro

        # Estimar pose de la cabeza (Pitch, Yaw, Roll)
        pose_data = None
        if face_results and face_results.multi_face_landmarks:
            pose_data = PoseEstimator.estimate_head_pose(frame_bgr.shape[:2], face_results.multi_face_landmarks[0])

        # 3. Lógica del "state machine" de la aplicación
        if self.app_state == "LISTENING":
            self._handle_listening_state(frame_bgr, face_results, pose_data)
        elif self.app_state == "IDENTIFYING":
            self._handle_identifying_state(frame_bgr, face_results)
        elif self.app_state == "AUTO_REGISTERING":
            self._handle_auto_registering_state(frame_bgr, face_results)
        elif self.app_state == "MONITORING":
            self._handle_monitoring_state(frame_bgr, face_results, hand_results, pose_results, pose_data)
        elif self.app_state == "REPOSO":
            # No se hace nada en modo reposo con el frame, solo se espera al MPU
            pass  # El MPUThread se encarga de cambiar el estado

        # 4. Dibujar interfaz de ROI y Overlays
        # Aquí se dibuja la interfaz de ROI sobre el frame original
        processed_frame_with_roi = roi_autoexp.draw_roi_interface(frame_bgr.copy())
        
        # Dibujar overlays de detección (malla facial, textos, alertas, etc.)
        final_display_frame = self._draw_frame_overlays(
            processed_frame_with_roi, face_results, hand_results, pose_results, pose_data, self.last_monitoring_data
        )
        
        # 5. Actualizar la visualización en la GUI
        if self.video_is_visible:
            try:
                h, w, ch = final_display_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(final_display_frame.data, w, h, bytes_per_line, QImage.Format_BGR888).rgbSwapped()
                self.camera_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                    self.camera_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
            except Exception as e:
                logger.error(f"Error al convertir y mostrar el fotograma final: {e}")
        
        return final_display_frame

    def closeEvent(self, event):
        """Asegura que todos los hilos se detengan al cerrar la aplicación."""
        logger.info("Cerrando la aplicación Copiloto-ID...")
        self.stop_camera()
        if self.mpu_thread and self.mpu_thread.isRunning(): 
            self.mpu_thread.stop()
            self.mpu_thread.wait()
        event.accept()               