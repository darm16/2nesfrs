# analytics_processor.py

from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Any

class AnalyticsProcessor:
    """
    Procesa una lista de eventos históricos para generar estadísticas y
    datos estructurados para visualización.
    """
    def __init__(self, events_data: List[Dict[str, Any]]):
        """
        Inicializa el procesador con los datos crudos de la base de datos.

        Args:
            events_data (List[Dict[str, Any]]): Una lista de diccionarios, donde cada 
                                               diccionario representa un evento de comportamiento.
        """
        self.events = events_data
        self.total_events = len(events_data) if events_data else 0

    def calculate_kpis(self) -> Dict[str, Any]:
        """
        Calcula los Indicadores Clave de Rendimiento (KPIs) a partir de los eventos.
        
        Devuelve:
            Un diccionario con las estadísticas clave para mostrar en la GUI.
        """
        if not self.events:
            return {
                "total_events": 0,
                "event_counts": {},
                "most_frequent_event": ("N/A", 0),
                "peak_hour": ("N/A", 0)
            }

        # Conteo de frecuencia de cada tipo de evento
        try:
            event_types = [event['tipo_evento'] for event in self.events if 'tipo_evento' in event]
            event_counts = Counter(event_types)
            most_frequent = event_counts.most_common(1)[0] if event_counts else ("N/A", 0)

            # Cálculo de la hora con más eventos
            hours = [datetime.fromisoformat(event['fecha_hora']).hour for event in self.events if 'fecha_hora' in event]
            hour_counts = Counter(hours)
            peak_hour = hour_counts.most_common(1)[0] if hour_counts else ("N/A", 0)

        except (ValueError, TypeError) as e:
            # Manejo de error si las fechas no tienen el formato esperado
            logger.error(f"Error al procesar fechas o tipos de evento en los datos: {e}")
            return {
                "total_events": self.total_events,
                "event_counts": {},
                "most_frequent_event": ("Error de formato", 0),
                "peak_hour": ("Error de formato", 0)
            }

        return {
            "total_events": self.total_events,
            "event_counts": dict(event_counts),
            "most_frequent_event": most_frequent,
            "peak_hour": peak_hour
        }

    def get_bar_chart_data(self, event_counts: Dict[str, int]) -> Tuple[List[str], List[int]]:
        """
        Prepara los datos para un gráfico de barras de frecuencia de eventos.
        
        Args:
            event_counts (Dict[str, int]): Diccionario con el conteo de cada evento.
        
        Devuelve:
            Una tupla con (lista_de_etiquetas_de_eventos, lista_de_valores_de_conteo).
        """
        if not event_counts:
            return [], []
            
        # Ordenar eventos por frecuencia para un gráfico más legible
        sorted_events = sorted(event_counts.items(), key=lambda item: item[1], reverse=True)
        
        labels = [item[0] for item in sorted_events]
        values = [item[1] for item in sorted_events]
        
        return labels, values

    def get_time_series_chart_data(self) -> Tuple[List[str], List[int]]:
        """
        Prepara los datos para un gráfico de línea de eventos distribuidos por hora.
        
        Devuelve:
            Una tupla con (lista_de_etiquetas_de_hora (0-23), lista_de_conteo_por_hora).
        """
        if not self.events:
            return [f"{h:02d}:00" for h in range(24)], [0] * 24

        try:
            hours = [datetime.fromisoformat(event['fecha_hora']).hour for event in self.events if 'fecha_hora' in event]
            hour_counts = Counter(hours)
        except (ValueError, TypeError) as e:
             logger.error(f"Error al procesar timestamps para el gráfico de tiempo: {e}")
             return [f"{h:02d}:00" for h in range(24)], [0] * 24
            
        # Crear una lista para las 24 horas del día, inicializada en 0
        time_series = [0] * 24
        for hour, count in hour_counts.items():
            if 0 <= hour < 24:
                time_series[hour] = count
                
        hour_labels = [f"{h:02d}:00" for h in range(24)]

        return hour_labels, time_series