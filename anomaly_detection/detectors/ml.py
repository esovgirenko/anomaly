"""Machine Learning based anomaly detection"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType


class MLDetector(AnomalyDetector):
    """
    ML-based anomaly detection using:
    - Isolation Forest
    - One-Class SVM
    """
    
    def __init__(
        self,
        name: str = "ml",
        enabled: bool = True,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        n_estimators: int = 100,
        online_learning: bool = False,
        min_samples: int = 50
    ):
        """
        Initialize ML detector
        
        Args:
            name: Detector name
            enabled: Whether detector is enabled
            method: Detection method ('isolation_forest' or 'one_class_svm')
            contamination: Expected proportion of anomalies (for Isolation Forest)
            n_estimators: Number of trees (for Isolation Forest)
            online_learning: Whether to retrain on new data
            min_samples: Minimum samples needed before detection
        """
        super().__init__(name, enabled)
        self.method = method
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.online_learning = online_learning
        self.min_samples = min_samples
        
        self._model = None
        self._scaler = StandardScaler()
        self._training_data: List[np.ndarray] = []
        self._feature_names: List[str] = []
        self._is_fitted = False
    
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """
        Train the ML model on historical data
        
        Args:
            data: List of metric dictionaries for training
        """
        if not data or len(data) < self.min_samples:
            return
        
        # Extract features
        feature_matrix, feature_names = self._extract_features(data)
        
        if feature_matrix.shape[0] < self.min_samples:
            return
        
        self._feature_names = feature_names
        
        # Scale features
        feature_matrix_scaled = self._scaler.fit_transform(feature_matrix)
        
        # Create and train model
        if self.method == "isolation_forest":
            self._model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == "one_class_svm":
            self._model = OneClassSVM(nu=self.contamination, kernel="rbf")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self._model.fit(feature_matrix_scaled)
        self._is_fitted = True
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect anomalies using ML model"""
        anomalies = []
        
        if not self._is_fitted:
            # Try to fit on historical data if available
            if historical_data and len(historical_data) >= self.min_samples:
                self.fit(historical_data)
            else:
                return anomalies
        
        # Extract features from current metrics
        feature_vector, feature_names = self._extract_features([metrics])
        
        if feature_vector.shape[1] != len(self._feature_names):
            # Feature mismatch - retrain if online learning enabled
            if self.online_learning and historical_data:
                self.fit(historical_data + [metrics])
                return self.detect(agent_id, metrics, historical_data)
            return anomalies
        
        # Scale features
        feature_vector_scaled = self._scaler.transform(feature_vector)
        
        # Predict
        prediction = self._model.predict(feature_vector_scaled)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # One-Class SVM returns -1 for anomalies, 1 for normal
        is_anomaly = prediction[0] == -1
        
        if is_anomaly:
            # Get anomaly score if available
            if hasattr(self._model, "decision_function"):
                score = self._model.decision_function(feature_vector_scaled)[0]
                # Normalize score to [0, 1] for severity
                # Lower scores = more anomalous
                severity = min(1.0, max(0.0, (1.0 - score) / 2.0))
            elif hasattr(self._model, "score_samples"):
                score = self._model.score_samples(feature_vector_scaled)[0]
                severity = min(1.0, max(0.0, -score / 10.0))
            else:
                severity = 0.7  # Default medium-high severity
            
            # Determine anomaly type based on which features are most unusual
            anomaly_type = self._determine_anomaly_type(metrics, feature_names, feature_vector[0])
            
            anomalies.append(Anomaly(
                agent_id=agent_id,
                anomaly_type=anomaly_type,
                severity=severity,
                timestamp=datetime.now(),
                description=f"ML anomaly detected using {self.method}: {len([v for v in metrics.values() if isinstance(v, (int, float))])} features analyzed",
                detector_name=self.name,
                metrics=metrics.copy(),
                metadata={
                    "method": self.method,
                    "prediction": int(prediction[0]),
                    "score": float(score) if 'score' in locals() else None,
                }
            ))
        
        # Online learning: update model with new data
        if self.online_learning and not is_anomaly:
            if historical_data:
                recent_data = historical_data[-self.min_samples:] + [metrics]
                if len(recent_data) >= self.min_samples:
                    self.fit(recent_data)
        
        return anomalies
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> tuple[np.ndarray, List[str]]:
        """
        Extract numeric features from metric dictionaries
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not data:
            return np.array([]).reshape(0, 0), []
        
        # Get all numeric keys from all data points
        all_keys = set()
        for d in data:
            for key, value in d.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    all_keys.add(key)
        
        # Remove non-feature keys
        feature_keys = sorted([k for k in all_keys if k not in ['timestamp', 'agent_id', 'error']])
        
        if not feature_keys:
            return np.array([]).reshape(len(data), 0), []
        
        # Build feature matrix
        feature_matrix = []
        for d in data:
            features = [float(d.get(k, 0.0)) for k in feature_keys]
            feature_matrix.append(features)
        
        return np.array(feature_matrix), feature_keys
    
    def _determine_anomaly_type(
        self,
        metrics: Dict[str, Any],
        feature_names: List[str],
        feature_values: np.ndarray
    ) -> AnomalyType:
        """Determine anomaly type based on which features are anomalous"""
        # Simple heuristic: check feature names
        performance_keywords = ['cpu', 'memory', 'response_time', 'latency', 'throughput']
        communication_keywords = ['messages', 'connections', 'requests', 'errors']
        temporal_keywords = ['duration', 'time', 'rate', 'frequency']
        
        for i, name in enumerate(feature_names):
            name_lower = name.lower()
            if any(kw in name_lower for kw in performance_keywords):
                return AnomalyType.PERFORMANCE
            if any(kw in name_lower for kw in communication_keywords):
                return AnomalyType.COMMUNICATION
            if any(kw in name_lower for kw in temporal_keywords):
                return AnomalyType.TEMPORAL
        
        return AnomalyType.BEHAVIORAL
    
    def reset(self) -> None:
        """Reset detector state"""
        self._model = None
        self._scaler = StandardScaler()
        self._training_data.clear()
        self._feature_names.clear()
        self._is_fitted = False

