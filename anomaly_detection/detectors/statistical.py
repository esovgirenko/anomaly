"""Statistical anomaly detection methods"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType


class StatisticalDetector(AnomalyDetector):
    """
    Statistical anomaly detection using:
    - Z-score analysis
    - Interquartile Range (IQR)
    - Moving average deviation
    """
    
    def __init__(
        self,
        name: str = "statistical",
        enabled: bool = True,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        window_size: int = 100,
        min_samples: int = 10
    ):
        """
        Initialize statistical detector
        
        Args:
            name: Detector name
            enabled: Whether detector is enabled
            z_score_threshold: Threshold for Z-score (default 3.0 = 3 standard deviations)
            iqr_multiplier: Multiplier for IQR method (default 1.5)
            window_size: Size of sliding window for moving average
            min_samples: Minimum samples needed before detection starts
        """
        super().__init__(name, enabled)
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.window_size = window_size
        self.min_samples = min_samples
        self._metric_history: Dict[str, List[float]] = {}
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies using statistical methods
        """
        anomalies = []
        
        # Extract numeric metrics
        numeric_metrics = self._extract_numeric_metrics(metrics)
        
        if not numeric_metrics:
            return anomalies
        
        # Update history
        for metric_name, value in numeric_metrics.items():
            if metric_name not in self._metric_history:
                self._metric_history[metric_name] = []
            
            self._metric_history[metric_name].append(value)
            
            # Keep only recent history
            if len(self._metric_history[metric_name]) > self.window_size:
                self._metric_history[metric_name].pop(0)
        
        # Need minimum samples before detecting
        if len(self._metric_history) == 0:
            return anomalies
        
        # Detect anomalies for each metric
        for metric_name, current_value in numeric_metrics.items():
            history = self._metric_history.get(metric_name, [])
            
            if len(history) < self.min_samples:
                continue
            
            # Use history up to but not including current value
            historical_values = np.array(history[:-1]) if len(history) > 1 else np.array(history)
            
            if len(historical_values) < self.min_samples:
                continue
            
            # Z-score detection
            z_score_anomaly = self._detect_z_score(
                agent_id, metric_name, current_value, historical_values
            )
            if z_score_anomaly:
                anomalies.append(z_score_anomaly)
            
            # IQR detection
            iqr_anomaly = self._detect_iqr(
                agent_id, metric_name, current_value, historical_values
            )
            if iqr_anomaly:
                anomalies.append(iqr_anomaly)
            
            # Moving average deviation
            ma_anomaly = self._detect_moving_average(
                agent_id, metric_name, current_value, historical_values
            )
            if ma_anomaly:
                anomalies.append(ma_anomaly)
        
        return anomalies
    
    def _extract_numeric_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from dictionary"""
        numeric = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key] = float(value)
        return numeric
    
    def _detect_z_score(
        self,
        agent_id: str,
        metric_name: str,
        current_value: float,
        historical_values: np.ndarray
    ) -> Optional[Anomaly]:
        """Detect anomaly using Z-score"""
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        if std == 0:
            return None
        
        z_score = abs((current_value - mean) / std)
        
        if z_score > self.z_score_threshold:
            severity = min(1.0, z_score / (self.z_score_threshold * 2))
            return Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.PERFORMANCE,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Z-score anomaly in {metric_name}: z={z_score:.2f}, value={current_value:.2f}, mean={mean:.2f}",
                detector_name=self.name,
                metrics={metric_name: current_value},
                metadata={
                    "method": "z_score",
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                }
            )
        
        return None
    
    def _detect_iqr(
        self,
        agent_id: str,
        metric_name: str,
        current_value: float,
        historical_values: np.ndarray
    ) -> Optional[Anomaly]:
        """Detect anomaly using Interquartile Range"""
        q1 = np.percentile(historical_values, 25)
        q3 = np.percentile(historical_values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return None
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        is_anomaly = current_value < lower_bound or current_value > upper_bound
        
        if is_anomaly:
            # Calculate severity based on how far outside bounds
            deviation = max(
                (lower_bound - current_value) / iqr if current_value < lower_bound else 0,
                (current_value - upper_bound) / iqr if current_value > upper_bound else 0
            )
            severity = min(1.0, 0.5 + deviation * 0.5)
            
            return Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.PERFORMANCE,
                severity=severity,
                timestamp=datetime.now(),
                description=f"IQR anomaly in {metric_name}: value={current_value:.2f}, bounds=[{lower_bound:.2f}, {upper_bound:.2f}]",
                detector_name=self.name,
                metrics={metric_name: current_value},
                metadata={
                    "method": "iqr",
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }
            )
        
        return None
    
    def _detect_moving_average(
        self,
        agent_id: str,
        metric_name: str,
        current_value: float,
        historical_values: np.ndarray
    ) -> Optional[Anomaly]:
        """Detect anomaly using moving average deviation"""
        if len(historical_values) < 20:
            return None
        
        # Calculate moving average over last window
        window = min(self.window_size, len(historical_values))
        ma = np.mean(historical_values[-window:])
        ma_std = np.std(historical_values[-window:])
        
        if ma_std == 0:
            return None
        
        deviation = abs(current_value - ma) / ma_std
        
        if deviation > self.z_score_threshold:
            severity = min(1.0, deviation / (self.z_score_threshold * 2))
            return Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.PERFORMANCE,
                severity=severity,
                timestamp=datetime.now(),
                description=f"Moving average anomaly in {metric_name}: value={current_value:.2f}, MA={ma:.2f}",
                detector_name=self.name,
                metrics={metric_name: current_value},
                metadata={
                    "method": "moving_average",
                    "moving_average": ma,
                    "deviation": deviation,
                }
            )
        
        return None
    
    def reset(self) -> None:
        """Reset detector state"""
        self._metric_history.clear()

