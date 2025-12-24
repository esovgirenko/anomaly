"""Prometheus metrics export"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Optional

from anomaly_detection.system.system import AnomalyDetectionSystem


# Prometheus metrics
anomalies_detected = Counter(
    'anomalies_detected_total',
    'Total number of anomalies detected',
    ['agent_id', 'anomaly_type', 'detector_name']
)

anomaly_severity = Histogram(
    'anomaly_severity',
    'Anomaly severity distribution',
    ['agent_id', 'anomaly_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

agents_monitored = Gauge(
    'agents_monitored',
    'Number of agents being monitored'
)

active_anomalies = Gauge(
    'active_anomalies',
    'Number of active anomalies',
    ['agent_id', 'anomaly_type']
)


class PrometheusExporter:
    """Exports metrics to Prometheus format"""
    
    def __init__(self, system: AnomalyDetectionSystem):
        self.system = system
    
    def record_anomaly(self, anomaly) -> None:
        """Record an anomaly in Prometheus metrics"""
        anomalies_detected.labels(
            agent_id=anomaly.agent_id,
            anomaly_type=anomaly.anomaly_type.value,
            detector_name=anomaly.detector_name
        ).inc()
        
        anomaly_severity.labels(
            agent_id=anomaly.agent_id,
            anomaly_type=anomaly.anomaly_type.value
        ).observe(anomaly.severity)
    
    def update_metrics(self) -> None:
        """Update gauge metrics"""
        agents_monitored.set(len(self.system.list_agents()))
        
        # Update active anomalies (last 1 hour)
        stats = self.system.get_overall_stats(hours=1)
        
        for agent_id in self.system.list_agents():
            agent_stats = self.system.get_agent_stats(agent_id, hours=1)
            for anomaly_type, count in agent_stats.get("by_type", {}).items():
                active_anomalies.labels(
                    agent_id=agent_id,
                    anomaly_type=anomaly_type
                ).set(count)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format"""
        self.update_metrics()
        return generate_latest()


def setup_prometheus_endpoint(app, system: AnomalyDetectionSystem):
    """Setup Prometheus endpoint on FastAPI app"""
    exporter = PrometheusExporter(system)
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return exporter.get_metrics()

