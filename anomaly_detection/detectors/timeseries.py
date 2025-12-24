"""Time series based anomaly detection"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType


class TimeSeriesDetector(AnomalyDetector):
    """
    Time series anomaly detection using:
    - Trend analysis
    - Seasonal pattern detection
    - Prophet forecasting (if available)
    - Simple moving average with deviation
    """
    
    def __init__(
        self,
        name: str = "timeseries",
        enabled: bool = True,
        window_size: int = 100,
        min_samples: int = 30,
        use_prophet: bool = True,
        trend_threshold: float = 2.0,
        seasonal_periods: Optional[List[int]] = None
    ):
        """
        Initialize time series detector
        
        Args:
            name: Detector name
            enabled: Whether detector is enabled
            window_size: Size of sliding window
            min_samples: Minimum samples needed
            use_prophet: Whether to use Prophet (requires prophet package)
            trend_threshold: Threshold for trend deviation (in standard deviations)
            seasonal_periods: List of seasonal periods to detect (e.g., [24, 168] for daily/weekly)
        """
        super().__init__(name, enabled)
        self.window_size = window_size
        self.min_samples = min_samples
        self.use_prophet = use_prophet and PROPHET_AVAILABLE
        self.trend_threshold = trend_threshold
        self.seasonal_periods = seasonal_periods or []
        
        self._time_series: Dict[str, List[tuple]] = {}
        self._prophet_models: Dict[str, Optional[Prophet]] = {}
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect time series anomalies"""
        anomalies = []
        timestamp = datetime.now()
        
        # Extract numeric metrics
        numeric_metrics = self._extract_numeric_metrics(metrics)
        
        if not numeric_metrics:
            return anomalies
        
        # Update time series for each metric
        for metric_name, value in numeric_metrics.items():
            if metric_name not in self._time_series:
                self._time_series[metric_name] = []
            
            self._time_series[metric_name].append((timestamp, value))
            
            # Keep only recent history
            if len(self._time_series[metric_name]) > self.window_size:
                self._time_series[metric_name].pop(0)
        
        # Detect anomalies for each metric
        for metric_name, current_value in numeric_metrics.items():
            time_series = self._time_series.get(metric_name, [])
            
            if len(time_series) < self.min_samples:
                continue
            
            # Extract values and timestamps
            timestamps, values = zip(*time_series)
            values_array = np.array(values)
            
            # Trend-based detection
            trend_anomaly = self._detect_trend(
                agent_id, metric_name, current_value, values_array, timestamp
            )
            if trend_anomaly:
                anomalies.append(trend_anomaly)
            
            # Prophet-based detection (if available)
            if self.use_prophet and len(time_series) >= self.min_samples:
                prophet_anomaly = self._detect_prophet(
                    agent_id, metric_name, time_series, timestamp
                )
                if prophet_anomaly:
                    anomalies.append(prophet_anomaly)
            
            # Seasonal anomaly detection
            if len(values_array) >= max(self.seasonal_periods) * 2 if self.seasonal_periods else 0:
                seasonal_anomaly = self._detect_seasonal(
                    agent_id, metric_name, values_array, timestamp
                )
                if seasonal_anomaly:
                    anomalies.append(seasonal_anomaly)
        
        return anomalies
    
    def _extract_numeric_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from dictionary"""
        numeric = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key] = float(value)
        return numeric
    
    def _detect_trend(
        self,
        agent_id: str,
        metric_name: str,
        current_value: float,
        values: np.ndarray,
        timestamp: datetime
    ) -> Optional[Anomaly]:
        """Detect anomalies based on trend deviation"""
        if len(values) < 10:
            return None
        
        # Calculate linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend_line = np.polyval(coeffs, x)
        
        # Calculate residuals
        residuals = values - trend_line
        std_residual = np.std(residuals)
        
        if std_residual == 0:
            return None
        
        # Expected value based on trend
        expected_value = np.polyval(coeffs, len(values))
        deviation = abs(current_value - expected_value) / std_residual
        
        if deviation > self.trend_threshold:
            severity = min(1.0, 0.5 + (deviation / (self.trend_threshold * 2)))
            return Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.TEMPORAL,
                severity=severity,
                timestamp=timestamp,
                description=f"Trend anomaly in {metric_name}: value={current_value:.2f}, expected={expected_value:.2f}, deviation={deviation:.2f}σ",
                detector_name=self.name,
                metrics={metric_name: current_value},
                metadata={
                    "method": "trend",
                    "trend_slope": float(coeffs[0]),
                    "expected_value": float(expected_value),
                    "deviation": float(deviation),
                }
            )
        
        return None
    
    def _detect_prophet(
        self,
        agent_id: str,
        metric_name: str,
        time_series: List[tuple],
        timestamp: datetime
    ) -> Optional[Anomaly]:
        """Detect anomalies using Prophet forecasting"""
        if not PROPHET_AVAILABLE or not self.use_prophet:
            return None
        
        try:
            # Prepare data for Prophet
            df = pd.DataFrame(time_series, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Use last value as current (excluding it from training)
            current_value = df.iloc[-1]['y']
            df_train = df.iloc[:-1].copy()
            
            if len(df_train) < self.min_samples:
                return None
            
            # Create and fit model
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=len(df_train) >= 14,
                daily_seasonality=len(df_train) >= 48
            )
            
            # Add custom seasonalities if specified
            for period in self.seasonal_periods:
                if len(df_train) >= period * 2:
                    model.add_seasonality(
                        name=f'custom_{period}',
                        period=period,
                        fourier_order=5
                    )
            
            model.fit(df_train)
            
            # Forecast for current timestamp
            future = pd.DataFrame({'ds': [timestamp]})
            forecast = model.predict(future)
            
            yhat = forecast['yhat'].iloc[0]
            yhat_lower = forecast['yhat_lower'].iloc[0]
            yhat_upper = forecast['yhat_upper'].iloc[0]
            
            # Check if current value is outside prediction interval
            is_anomaly = current_value < yhat_lower or current_value > yhat_upper
            
            if is_anomaly:
                # Calculate severity based on how far outside interval
                if current_value < yhat_lower:
                    deviation = (yhat_lower - current_value) / (yhat_upper - yhat_lower)
                else:
                    deviation = (current_value - yhat_upper) / (yhat_upper - yhat_lower)
                
                severity = min(1.0, 0.5 + deviation * 0.5)
                
                return Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=severity,
                    timestamp=timestamp,
                    description=f"Prophet forecast anomaly in {metric_name}: value={current_value:.2f}, forecast={yhat:.2f} [{yhat_lower:.2f}, {yhat_upper:.2f}]",
                    detector_name=self.name,
                    metrics={metric_name: current_value},
                    metadata={
                        "method": "prophet",
                        "forecast": float(yhat),
                        "forecast_lower": float(yhat_lower),
                        "forecast_upper": float(yhat_upper),
                    }
                )
        except Exception as e:
            # Prophet can fail on various edge cases, fail silently
            pass
        
        return None
    
    def _detect_seasonal(
        self,
        agent_id: str,
        metric_name: str,
        values: np.ndarray,
        timestamp: datetime
    ) -> Optional[Anomaly]:
        """Detect seasonal pattern anomalies"""
        if not self.seasonal_periods:
            return None
        
        for period in self.seasonal_periods:
            if len(values) < period * 2:
                continue
            
            # Extract seasonal component
            seasonal_values = values[-period * 2:-period]
            current_seasonal_position = len(values) % period
            
            if len(seasonal_values) >= period:
                # Compare current value with same position in previous season
                expected_value = np.mean(seasonal_values[-period:])
                std_value = np.std(seasonal_values[-period:])
                
                if std_value == 0:
                    continue
                
                deviation = abs(values[-1] - expected_value) / std_value
                
                if deviation > self.trend_threshold:
                    severity = min(1.0, 0.5 + (deviation / (self.trend_threshold * 2)))
                    return Anomaly(
                        agent_id=agent_id,
                        anomaly_type=AnomalyType.TEMPORAL,
                        severity=severity,
                        timestamp=timestamp,
                        description=f"Seasonal anomaly in {metric_name} (period={period}): value={values[-1]:.2f}, expected={expected_value:.2f}",
                        detector_name=self.name,
                        metrics={metric_name: float(values[-1])},
                        metadata={
                            "method": "seasonal",
                            "period": period,
                            "expected_value": float(expected_value),
                            "deviation": float(deviation),
                        }
                    )
        
        return None
    
    def reset(self) -> None:
        """Reset detector state"""
        self._time_series.clear()
        self._prophet_models.clear()

