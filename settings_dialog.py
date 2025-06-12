# settings_dialog.py

import os
import copy
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout, 
                             QDialogButtonBox, QSpinBox, QDoubleSpinBox, QCheckBox, 
                             QComboBox, QSlider, QLabel, QGroupBox, QPushButton, 
                             QLineEdit, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from config_manager import ConfigManager
from logging_setup import logger

class SettingsDialog(QDialog):
    """
    Ventana de diálogo para editar la configuración completa de la aplicación.
    """
    settings_saved = pyqtSignal()

    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración del Sistema Copiloto-ID")
        self.setMinimumSize(700, 650)
        
        self.config = current_config
        self.widgets = {}

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Crear y añadir cada pestaña
        self.tabs.addTab(self._create_fatigue_tab(), "Detección de Fatiga")
        self.tabs.addTab(self._create_recognition_tab(), "Reconocimiento Facial")
        self.tabs.addTab(self._create_voice_alerts_tab(), "Alertas por Voz")
        self.tabs.addTab(self._create_camera_tab(), "Cámara y Vídeo")
        self.tabs.addTab(self._create_advanced_tab(), "Avanzado y Seguridad")
        
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Save).setText("Guardar Cambios")
        button_box.button(QDialogButtonBox.Cancel).setText("Cancelar")
        button_box.accepted.connect(self._save_settings)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self._load_settings()

    def _create_fatigue_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        self.widgets['fatigue_detection_thresholds'] = {}
        self.widgets['event_toggles'] = {}

        thresholds_group = QGroupBox("Umbrales de Detección")
        form_layout = QFormLayout(thresholds_group)
        
        # Crear los widgets primero sin el rango
        fatigue_widgets_map = {
            "EAR_THRESHOLD": (QDoubleSpinBox(decimals=2, singleStep=0.01), "Proporción de Aspecto del Ojo (EAR) mínima antes de considerar somnolencia."),
            "MOE_THRESHOLD": (QDoubleSpinBox(decimals=2, singleStep=0.01), "Ratio Boca/Ojo (MOE). Umbral para confirmar bostezos."),
            "YAWN_FRAMES_THRESHOLD": (QSpinBox(suffix=" frames"), "Nº de fotogramas con MAR alto para confirmar un bostezo."),
            "DISTRACTION_FRAMES_THRESHOLD": (QSpinBox(suffix=" frames"), "Nº de fotogramas con el rostro ausente para considerarlo distracción."),
            "STATIC_POSITION_SECONDS": (QSpinBox(suffix=" seg"), "Tiempo en la misma posición antes de sugerir un cambio."),
            "EYE_RUBBING_FRAMES": (QSpinBox(suffix=" frames"), "Nº de fotogramas frotándose los ojos para generar una alerta.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in fatigue_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['fatigue_detection_thresholds'][key] = widget
            form_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
        
        # Establecer los rangos después de crear los widgets y agregarlos al diccionario
        self.widgets['fatigue_detection_thresholds']['EAR_THRESHOLD'].setRange(0.1, 0.5)
        self.widgets['fatigue_detection_thresholds']['MOE_THRESHOLD'].setRange(0.1, 1.0)
        self.widgets['fatigue_detection_thresholds']['YAWN_FRAMES_THRESHOLD'].setRange(5, 100)
        self.widgets['fatigue_detection_thresholds']['DISTRACTION_FRAMES_THRESHOLD'].setRange(10, 300)
        self.widgets['fatigue_detection_thresholds']['STATIC_POSITION_SECONDS'].setRange(600, 7200)
        self.widgets['fatigue_detection_thresholds']['EYE_RUBBING_FRAMES'].setRange(5, 60)
        layout.addWidget(thresholds_group)

        events_group = QGroupBox("Activación de Detecciones")
        events_layout = QVBoxLayout(events_group)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content)
        
        event_keys = sorted(self.config.get('event_toggles', {}).keys())
        for key in event_keys:
            checkbox = QCheckBox(key)
            checkbox.setToolTip(f"Activa o desactiva la detección y alerta para el evento '{key}'.")
            self.widgets['event_toggles'][key] = checkbox
            scroll_layout.addWidget(checkbox)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        events_layout.addWidget(scroll_area)
        layout.addWidget(events_group)
        return tab_widget

    def _create_recognition_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        self.widgets['recognition_settings'] = {}
        
        group = QGroupBox("Ajustes de Reconocimiento")
        form_layout = QFormLayout(group)
        
        self.widgets['recognition_settings']['threshold'] = QDoubleSpinBox()
        self.widgets['recognition_settings']['threshold'].setDecimals(2); self.widgets['recognition_settings']['threshold'].setSingleStep(0.01); self.widgets['recognition_settings']['threshold'].setRange(0.1, 0.95)
        self.widgets['recognition_settings']['threshold'].setToolTip("Nivel de confianza mínimo para un reconocimiento facial exitoso.")
        form_layout.addRow("Umbral de Similitud:", self.widgets['recognition_settings']['threshold'])

        self.widgets['recognition_settings']['mode'] = QComboBox(); self.widgets['recognition_settings']['mode'].addItems(["simple", "multi-pose"])
        self.widgets['recognition_settings']['mode'].setToolTip("'Simple': 1 embedding por usuario (rápido).\n'Multi-pose': Varios embeddings (más robusto).")
        form_layout.addRow("Modo de Reconocimiento:", self.widgets['recognition_settings']['mode'])
        layout.addWidget(group)
        
        pose_group = QGroupBox("Límites de Pose Aceptable (en grados)")
        pose_layout = QFormLayout(pose_group)
        self.widgets['recognition_settings']['pose_limits'] = {}
        # Crear los widgets primero sin el rango
        pose_widgets_map = {
            "MAX_ABS_PITCH": (QSpinBox(), "Máxima inclinación vertical (arriba/abajo)."),
            "MAX_ABS_YAW": (QSpinBox(), "Máximo giro horizontal (izquierda/derecha)."),
            "MAX_ABS_ROLL": (QSpinBox(), "Máxima inclinación lateral.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in pose_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['recognition_settings']['pose_limits'][key] = widget
            pose_layout.addRow(f"{key.replace('MAX_ABS_', '')}:", widget)
        
        # Establecer los rangos después de crear los widgets
        self.widgets['recognition_settings']['pose_limits']['MAX_ABS_PITCH'].setRange(5, 45)
        self.widgets['recognition_settings']['pose_limits']['MAX_ABS_YAW'].setRange(5, 45)
        self.widgets['recognition_settings']['pose_limits']['MAX_ABS_ROLL'].setRange(5, 45)
        layout.addWidget(pose_group)
        layout.addStretch()
        return tab_widget

    def _create_voice_alerts_tab(self):
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(); alerts_layout = QVBoxLayout(scroll_content)
        self.widgets['voice_alerts'] = {}
        alert_keys = sorted(self.config.get('voice_alerts', {}).keys())

        for key in alert_keys:
            group = QGroupBox(f"Alerta: {key}")
            form_layout = QFormLayout(group)
            enabled_check = QCheckBox("Habilitada"); enabled_check.setToolTip(f"Activa o desactiva la alerta de voz para '{key}'.")
            text_edit = QLineEdit(); text_edit.setToolTip(f"La frase que el sistema dirá para la alerta '{key}'.")
            priority_combo = QComboBox(); priority_combo.addItems(["Crítica (0)", "Alta (1)", "Media (2)", "Baja (3)"]); priority_combo.setToolTip("Define la urgencia de la alerta.\n'Crítica (0)' interrumpe otros mensajes.")
            self.widgets['voice_alerts'][key] = {'enabled': enabled_check, 'text': text_edit, 'priority': priority_combo}
            form_layout.addRow(enabled_check)
            form_layout.addRow("Texto:", text_edit)
            form_layout.addRow("Prioridad:", priority_combo)
            alerts_layout.addWidget(group)
        
        alerts_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        return tab_widget
        
    def _create_camera_tab(self):
        tab_widget = QWidget()
        layout = QFormLayout(tab_widget)
        self.widgets['camera_settings'] = {}
        
        # Crear los widgets primero sin el rango
        cam_widgets_map = {
            "camera_index": (QSpinBox(), "Índice de la cámara a utilizar (-1 para ninguna)."),
            "FRAME_WIDTH": (QSpinBox(), "Ancho de la resolución de captura. Puede requerir reinicio."),
            "FRAME_HEIGHT": (QSpinBox(), "Alto de la resolución de captura. Puede requerir reinicio."),
            "MAX_CAMERA_INDEX_TO_CHECK": (QSpinBox(), "Hasta qué índice buscar cámaras al iniciar.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in cam_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['camera_settings'][key] = widget
            layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
        
        # Establecer los rangos después de crear los widgets
        self.widgets['camera_settings']['camera_index'].setRange(-1, 10)
        self.widgets['camera_settings']['FRAME_WIDTH'].setRange(320, 1920)
        self.widgets['camera_settings']['FRAME_HEIGHT'].setRange(240, 1080)
        self.widgets['camera_settings']['MAX_CAMERA_INDEX_TO_CHECK'].setRange(0, 10)
        
        return tab_widget
        
    def _create_advanced_tab(self):
        tab_widget = QWidget(); layout = QVBoxLayout(tab_widget)
        self.widgets['privacy_settings'] = {}; self.widgets['state_machine_thresholds'] = {}
        self.widgets['profile_enrichment'] = {}; self.widgets['hardware_settings'] = {}
        self.widgets['pausa_activa_settings'] = {}

        sec_group = QGroupBox("Seguridad y Privacidad"); sec_layout = QFormLayout(sec_group)
        self.widgets['privacy_settings']['privacy_mode_default'] = QCheckBox("Activar Modo Privacidad por defecto")
        self.widgets['privacy_settings']['privacy_mode_default'].setToolTip("Si está marcado, la aplicación iniciará con las fotos ocultas.")
        sec_layout.addRow(self.widgets['privacy_settings']['privacy_mode_default'])
        change_pass_button = QPushButton("Cambiar Clave de Administrador..."); change_pass_button.setToolTip("Abre un diálogo para cambiar la clave que protege ajustes y fotos.")
        change_pass_button.clicked.connect(self._handle_change_password)
        sec_layout.addRow(change_pass_button)
        layout.addWidget(sec_group)

        state_group = QGroupBox("Umbrales del Flujo Automático"); state_layout = QFormLayout(state_group)
        
        # Crear los widgets primero sin el rango
        state_widgets_map = {
            "stable_face_frames_to_identify": (QSpinBox(suffix=" frames"), "Nº de frames estables para iniciar la identificación."),
            "face_lost_frames_to_logout": (QSpinBox(suffix=" frames"), "Nº de frames sin rostro para cerrar la sesión."),
            "auto_register_seconds": (QSpinBox(suffix=" seg"), "Tiempo con rostro desconocido antes de registrarlo.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in state_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['state_machine_thresholds'][key] = widget
            state_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
        
        # Establecer los rangos después de crear los widgets
        self.widgets['state_machine_thresholds']['stable_face_frames_to_identify'].setRange(30, 300)
        self.widgets['state_machine_thresholds']['face_lost_frames_to_logout'].setRange(100, 1800)
        self.widgets['state_machine_thresholds']['auto_register_seconds'].setRange(5, 60)
        
        layout.addWidget(state_group)
        
        enrich_group = QGroupBox("Enriquecimiento de Perfil Dinámico"); enrich_layout = QFormLayout(enrich_group)
        self.widgets['profile_enrichment']['enabled'] = QCheckBox("Habilitar"); self.widgets['profile_enrichment']['enabled'].setToolTip("Permite capturar nuevos embeddings durante el primer minuto de monitoreo.")
        enrich_layout.addRow(self.widgets['profile_enrichment']['enabled'])
        # Crear los widgets primero sin el rango
        enrich_widgets_map = {
            "embeddings_per_session_target": (QSpinBox(), "Nº máximo de nuevos embeddings a capturar por sesión."),
            "min_pose_difference_threshold": (QDoubleSpinBox(decimals=1), "Diferencia de pose mínima para capturar un nuevo embedding.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in enrich_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['profile_enrichment'][key] = widget
            enrich_layout.addRow(f"{key.replace('_', ' ').title()}:", widget)
        
        # Establecer los rangos después de crear los widgets
        self.widgets['profile_enrichment']['embeddings_per_session_target'].setRange(1, 20)
        self.widgets['profile_enrichment']['min_pose_difference_threshold'].setRange(5.0, 30.0)
        layout.addWidget(enrich_group)
        
        hw_group = QGroupBox("Hardware (MPU6050)"); hw_layout = QFormLayout(hw_group)
        self.widgets['hardware_settings']['USE_MPU'] = QCheckBox("Habilitar sensor de movimiento MPU6050")
        self.widgets['hardware_settings']['USE_MPU'].setToolTip("Activa el modo de reposo automático si se detecta el sensor.")
        hw_layout.addRow(self.widgets['hardware_settings']['USE_MPU'])
        # Crear los widgets primero sin el rango
        hw_widgets_map = {
            "mpu_sleep_threshold_minutes": (QSpinBox(suffix=" min"), "Minutos de inactividad para entrar en modo reposo."),
            "mpu_accel_threshold": (QDoubleSpinBox(decimals=2), "Umbral del acelerómetro para detectar movimiento."),
            "mpu_gyro_threshold": (QDoubleSpinBox(decimals=2), "Umbral del giroscopio para detectar movimiento.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in hw_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['hardware_settings'][key] = widget
            hw_layout.addRow(f"{key.replace('mpu_', '').replace('_', ' ').title()}:", widget)
        
        # Establecer los rangos después de crear los widgets
        self.widgets['hardware_settings']['mpu_sleep_threshold_minutes'].setRange(1, 60)
        self.widgets['hardware_settings']['mpu_accel_threshold'].setRange(0.0, 5.0)
        self.widgets['hardware_settings']['mpu_gyro_threshold'].setRange(0.0, 5.0)
        layout.addWidget(hw_group)

        pa_group = QGroupBox("Pausa Activa"); pa_layout = QFormLayout(pa_group)
        
        # Crear los widgets primero sin el rango
        pa_widgets_map = {
             "work_duration_seconds": (QSpinBox(suffix=" seg"), "Tiempo de trabajo continuo antes de sugerir una pausa."),
             "reset_threshold_seconds": (QSpinBox(suffix=" seg"), "Tiempo de ausencia para resetear el contador de trabajo.")
        }
        
        # Agregar los widgets al diccionario y al layout
        for key, (widget, tooltip) in pa_widgets_map.items():
            widget.setToolTip(tooltip)
            self.widgets['pausa_activa_settings'][key] = widget
            pa_layout.addRow(f"{key.replace('_', ' ').replace('seconds', 'seg').title()}:", widget)
        
        # Establecer los rangos después de crear los widgets
        self.widgets['pausa_activa_settings']['work_duration_seconds'].setRange(600, 7200)
        self.widgets['pausa_activa_settings']['reset_threshold_seconds'].setRange(60, 600)
        layout.addWidget(pa_group)

        layout.addStretch()
        return tab_widget

    def _load_settings(self):
        logger.info("Cargando configuración actual en el diálogo de ajustes.")
        try:
            for section_key, section_widgets in self.widgets.items():
                config_section = self.config.get(section_key, {})
                if section_key == 'voice_alerts':
                    for key, controls in section_widgets.items():
                        alert_config = config_section.get(key, {})
                        controls['enabled'].setChecked(alert_config.get('enabled', True))
                        controls['text'].setText(alert_config.get('text', ''))
                        controls['priority'].setCurrentIndex(alert_config.get('priority', 2))
                else:
                    for key, widget in section_widgets.items():
                        if isinstance(config_section.get(key), dict):
                            for sub_key, sub_widget in widget.items():
                                sub_widget.setValue(config_section[key].get(sub_key, 0))
                        elif isinstance(widget, QCheckBox):
                            widget.setChecked(config_section.get(key, False))
                        elif isinstance(widget, QComboBox):
                            widget.setCurrentText(str(config_section.get(key, "")))
                        else:
                            widget.setValue(config_section.get(key, 0))
        except Exception as e:
            logger.error(f"Error al cargar los ajustes en la GUI: {e}", exc_info=True)

    def _save_settings(self):
        logger.info("Guardando nuevos ajustes desde el diálogo.")
        new_config = copy.deepcopy(self.config)
        try:
            for section_key, section_widgets in self.widgets.items():
                config_section = new_config.get(section_key, {})
                if section_key == 'voice_alerts':
                    for key, controls in section_widgets.items():
                        if key not in config_section: config_section[key] = {}
                        config_section[key]['enabled'] = controls['enabled'].isChecked()
                        config_section[key]['text'] = controls['text'].text()
                        config_section[key]['priority'] = controls['priority'].currentIndex()
                else:
                    for key, widget in section_widgets.items():
                         if isinstance(config_section.get(key), dict):
                            if key not in config_section: config_section[key] = {}
                            for sub_key, sub_widget in widget.items():
                                config_section[key][sub_key] = sub_widget.value()
                         elif isinstance(widget, QCheckBox):
                            config_section[key] = widget.isChecked()
                         elif isinstance(widget, QComboBox):
                            config_section[key] = widget.currentText()
                         else:
                            config_section[key] = widget.value()
            
            ConfigManager.save_full_config(new_config)
            QMessageBox.information(self, "Éxito", "La configuración ha sido guardada.\nAlgunos cambios pueden requerir un reinicio.")
            self.settings_saved.emit()
            self.accept()
        except Exception as e:
            logger.error(f"Error al guardar la configuración: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"No se pudo guardar la configuración:\n{e}")

    def _handle_change_password(self):
        current_pass, ok1 = QInputDialog.getText(self, "Verificación", "Introduzca su clave actual:", QLineEdit.Password)
        if not ok1 or not current_pass: return
        if not ConfigManager.verify_password(current_pass):
            QMessageBox.warning(self, "Error", "La clave actual es incorrecta.")
            return
        
        new_pass, ok2 = QInputDialog.getText(self, "Nueva Clave", "Introduzca la nueva clave:", QLineEdit.Password)
        if not ok2 or not new_pass: return
        if len(new_pass) < 4:
            QMessageBox.warning(self, "Error", "La nueva clave debe tener al menos 4 caracteres.")
            return
            
        confirm_pass, ok3 = QInputDialog.getText(self, "Confirmar Clave", "Confirme la nueva clave:", QLineEdit.Password)
        if not ok3 or new_pass != confirm_pass:
            QMessageBox.warning(self, "Error", "Las nuevas claves no coinciden.")
            return
            
        recovery_code = ConfigManager.set_new_password(new_pass)
        QMessageBox.information(self, "Éxito", f"La clave ha sido cambiada.\n\n**GUARDE ESTE CÓDIGO DE RECUPERACIÓN EN UN LUGAR SEGURO:**\n\n{recovery_code}\n\nEs la única forma de recuperar el acceso si olvida su nueva clave.")