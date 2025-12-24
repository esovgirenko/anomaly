"""Main anomaly detection system"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml

from anomaly_detection.core.monitor import Agent, AgentMonitor
from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType
from anomaly_detection.management.registry import AnomalyRegistry
from anomaly_detection.management.alert_manager import AlertManager
from anomaly_detection.management.metrics_collector import MetricsCollector

from anomaly_detection.detectors.statistical import StatisticalDetector
from anomaly_detection.detectors.ml import MLDetector
from anomaly_detection.detectors.rule_based import RuleBasedDetector, Rule
from anomaly_detection.detectors.timeseries import TimeSeriesDetector


class AnomalyDetectionSystem:
    """
    Main system for anomaly detection in multi-agent systems
    
    Coordinates:
    - Agent monitoring
    - Anomaly detection
    - Alert management
    - Metrics collection
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize anomaly detection system
        
        Args:
            config_path: Path to configuration YAML file
            config: Configuration dictionary (alternative to config_path)
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            self.config = {}
        
        # Initialize components
        self.registry = AnomalyRegistry(
            deduplication_window=timedelta(
                minutes=self.config.get("deduplication_window_minutes", 5)
            )
        )
        self.alert_manager = AlertManager()
        self.metrics_collector = MetricsCollector()
        
        # Agent monitors
        self._agents: Dict[str, Agent] = {}
        self._monitors: Dict[str, AgentMonitor] = {}
        
        # Detectors
        self._detectors: List[AnomalyDetector] = []
        self._initialize_detectors()
        
        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._collection_interval = self.config.get("collection_interval_seconds", 10)
        self._prometheus_exporter = None  # Will be set if Prometheus is used
    
    def _initialize_detectors(self) -> None:
        """Initialize detectors from configuration"""
        detectors_config = self.config.get("detectors", {})
        
        # Statistical detector
        if detectors_config.get("statistical", {}).get("enabled", True):
            stat_config = detectors_config.get("statistical", {})
            self._detectors.append(StatisticalDetector(
                z_score_threshold=stat_config.get("z_score_threshold", 3.0),
                iqr_multiplier=stat_config.get("iqr_multiplier", 1.5),
                window_size=stat_config.get("window_size", 100),
            ))
        
        # ML detector
        if detectors_config.get("ml", {}).get("enabled", False):
            ml_config = detectors_config.get("ml", {})
            self._detectors.append(MLDetector(
                method=ml_config.get("method", "isolation_forest"),
                contamination=ml_config.get("contamination", 0.1),
                n_estimators=ml_config.get("n_estimators", 100),
                online_learning=ml_config.get("online_learning", False),
            ))
        
        # Rule-based detector
        if detectors_config.get("rule_based", {}).get("enabled", True):
            # Try to load rules from config
            rules_config = self.config.get("rules", [])
            # If not in main config, try loading from rules.yaml
            if not rules_config:
                try:
                    rules_path = self.config.get("rules_path", "config/rules.yaml")
                    import os
                    if os.path.exists(rules_path):
                        with open(rules_path, 'r') as f:
                            rules_yaml = yaml.safe_load(f)
                            rules_config = rules_yaml.get("rules", [])
                except Exception as e:
                    print(f"Warning: Could not load rules from file: {e}")
            
            rule_detector = RuleBasedDetector()
            if rules_config:
                rule_detector.load_rules_from_dict(rules_config)
            self._detectors.append(rule_detector)
        
        # Time series detector
        if detectors_config.get("timeseries", {}).get("enabled", False):
            ts_config = detectors_config.get("timeseries", {})
            self._detectors.append(TimeSeriesDetector(
                window_size=ts_config.get("window_size", 100),
                use_prophet=ts_config.get("use_prophet", False),
            ))
    
    def register_agent(
        self,
        agent_id: str,
        metrics_endpoint: Optional[str] = None,
        metadata: Optional[Dict] = None,
        custom_detectors: Optional[List[AnomalyDetector]] = None
    ) -> Agent:
        """
        Register an agent for monitoring
        
        Args:
            agent_id: Unique agent identifier
            metrics_endpoint: Optional endpoint to fetch metrics from
            metadata: Optional metadata about the agent
            custom_detectors: Optional custom detectors for this agent
            
        Returns:
            Registered Agent instance
        """
        agent = Agent(agent_id, metrics_endpoint, metadata)
        self._agents[agent_id] = agent
        
        # Create monitor with detectors
        detectors = custom_detectors or self._detectors
        monitor = AgentMonitor(
            agent=agent,
            detectors=detectors,
            metrics_collector=self.metrics_collector.collect_from_agent
        )
        self._monitors[agent_id] = monitor
        
        return agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self._agents:
            del self._agents[agent_id]
        if agent_id in self._monitors:
            del self._monitors[agent_id]
    
    async def start_monitoring(self) -> None:
        """Start continuous monitoring of all registered agents"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
                # Monitor all agents
                tasks = [
                    self._monitor_agent(agent_id)
                    for agent_id in self._agents.keys()
                ]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait before next collection
                await asyncio.sleep(self._collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._collection_interval)
    
    async def _monitor_agent(self, agent_id: str) -> None:
        """Monitor a single agent"""
        monitor = self._monitors.get(agent_id)
        if not monitor:
            return
        
        try:
            # Detect anomalies
            anomalies = await monitor.detect_anomalies()
            
            # Register and alert on anomalies
            for anomaly in anomalies:
                if self.registry.register(anomaly):
                    # Only alert on newly registered (non-duplicate) anomalies
                    await self.alert_manager.send_alert(anomaly)
                    
        except Exception as e:
            print(f"Error monitoring agent {agent_id}: {e}")
    
    async def check_agent(self, agent_id: str) -> List[Anomaly]:
        """
        Manually trigger anomaly detection for an agent
        
        Returns:
            List of detected anomalies
        """
        monitor = self._monitors.get(agent_id)
        if not monitor:
            return []
        
        anomalies = await monitor.detect_anomalies()
        
        # Register anomalies
        for anomaly in anomalies:
            if self.registry.register(anomaly):
                await self.alert_manager.send_alert(anomaly)
        
        return anomalies
    
    def get_recent_anomalies(
        self,
        hours: int = 24,
        agent_id: Optional[str] = None,
        anomaly_type: Optional[AnomalyType] = None,
        min_severity: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Anomaly]:
        """Get recent anomalies"""
        return self.registry.get_recent_anomalies(
            hours=hours,
            agent_id=agent_id,
            anomaly_type=anomaly_type,
            min_severity=min_severity,
            limit=limit
        )
    
    def get_agent_stats(self, agent_id: str, hours: int = 24) -> Dict:
        """Get statistics for an agent"""
        return self.registry.get_agent_stats(agent_id, hours)
    
    def get_overall_stats(self, hours: int = 24) -> Dict:
        """Get overall system statistics"""
        return self.registry.get_overall_stats(hours)
    
    def list_agents(self) -> List[str]:
        """Get list of registered agent IDs"""
        return list(self._agents.keys())
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)
    
    def add_detector(self, detector: AnomalyDetector) -> None:
        """Add a custom detector to all agents"""
        self._detectors.append(detector)
        # Update existing monitors
        for monitor in self._monitors.values():
            monitor.detectors.append(detector)
    
    def __repr__(self) -> str:
        return f"AnomalyDetectionSystem(agents={len(self._agents)}, detectors={len(self._detectors)})"

