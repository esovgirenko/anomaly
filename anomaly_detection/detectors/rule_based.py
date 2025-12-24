"""Rule-based anomaly detection"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
from enum import Enum

from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType


class ConditionOperator(Enum):
    """Operators for rule conditions"""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NE = "!="
    CONTAINS = "contains"
    REGEX = "regex"


class RuleCondition:
    """Represents a single condition in a rule"""
    
    def __init__(
        self,
        metric: str,
        operator: ConditionOperator,
        value: Any,
        description: Optional[str] = None
    ):
        self.metric = metric
        self.operator = operator
        self.value = value
        self.description = description or f"{metric} {operator.value} {value}"
    
    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate condition against metrics"""
        metric_value = metrics.get(self.metric)
        
        if metric_value is None:
            return False
        
        try:
            if self.operator == ConditionOperator.GT:
                return float(metric_value) > float(self.value)
            elif self.operator == ConditionOperator.GTE:
                return float(metric_value) >= float(self.value)
            elif self.operator == ConditionOperator.LT:
                return float(metric_value) < float(self.value)
            elif self.operator == ConditionOperator.LTE:
                return float(metric_value) <= float(self.value)
            elif self.operator == ConditionOperator.EQ:
                return metric_value == self.value
            elif self.operator == ConditionOperator.NE:
                return metric_value != self.value
            elif self.operator == ConditionOperator.CONTAINS:
                return str(self.value) in str(metric_value)
            elif self.operator == ConditionOperator.REGEX:
                return bool(re.search(str(self.value), str(metric_value)))
        except (ValueError, TypeError):
            return False
        
        return False


class Rule:
    """Represents a detection rule"""
    
    def __init__(
        self,
        name: str,
        conditions: List[RuleCondition],
        anomaly_type: AnomalyType,
        severity: float,
        description: str,
        logic: str = "AND",  # "AND" or "OR"
        time_window: Optional[timedelta] = None,
        min_occurrences: int = 1
    ):
        self.name = name
        self.conditions = conditions
        self.anomaly_type = anomaly_type
        self.severity = severity
        self.description = description
        self.logic = logic.upper()
        self.time_window = time_window
        self.min_occurrences = min_occurrences
        self._occurrences: List[datetime] = []
    
    def evaluate(self, metrics: Dict[str, Any], timestamp: datetime) -> bool:
        """Evaluate rule against metrics"""
        # Evaluate conditions
        if self.logic == "AND":
            matches = all(cond.evaluate(metrics) for cond in self.conditions)
        else:  # OR
            matches = any(cond.evaluate(metrics) for cond in self.conditions)
        
        if matches:
            # Record occurrence
            self._occurrences.append(timestamp)
            
            # Clean old occurrences outside time window
            if self.time_window:
                cutoff = timestamp - self.time_window
                self._occurrences = [t for t in self._occurrences if t > cutoff]
            
            # Check if we have enough occurrences
            return len(self._occurrences) >= self.min_occurrences
        
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """Create rule from dictionary"""
        conditions = []
        for cond_data in data.get("conditions", []):
            operator = ConditionOperator(cond_data["operator"])
            condition = RuleCondition(
                metric=cond_data["metric"],
                operator=operator,
                value=cond_data["value"],
                description=cond_data.get("description")
            )
            conditions.append(condition)
        
        time_window = None
        if "time_window_seconds" in data:
            time_window = timedelta(seconds=data["time_window_seconds"])
        
        return cls(
            name=data["name"],
            conditions=conditions,
            anomaly_type=AnomalyType(data.get("anomaly_type", "behavioral")),
            severity=float(data.get("severity", 0.7)),
            description=data.get("description", ""),
            logic=data.get("logic", "AND"),
            time_window=time_window,
            min_occurrences=int(data.get("min_occurrences", 1))
        )


class RuleBasedDetector(AnomalyDetector):
    """
    Rule-based anomaly detection using configurable rules
    
    Rules can have multiple conditions with AND/OR logic,
    time windows, and minimum occurrence thresholds.
    """
    
    def __init__(
        self,
        name: str = "rule_based",
        enabled: bool = True,
        rules: Optional[List[Rule]] = None
    ):
        """
        Initialize rule-based detector
        
        Args:
            name: Detector name
            enabled: Whether detector is enabled
            rules: List of rules to use for detection
        """
        super().__init__(name, enabled)
        self.rules = rules or []
    
    def add_rule(self, rule: Rule) -> None:
        """Add a detection rule"""
        self.rules.append(rule)
    
    def load_rules_from_dict(self, rules_data: List[Dict[str, Any]]) -> None:
        """Load rules from dictionary format"""
        self.rules = [Rule.from_dict(rule_data) for rule_data in rules_data]
    
    def detect(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> List[Anomaly]:
        """Detect anomalies using configured rules"""
        anomalies = []
        timestamp = datetime.now()
        
        for rule in self.rules:
            if rule.evaluate(metrics, timestamp):
                anomalies.append(Anomaly(
                    agent_id=agent_id,
                    anomaly_type=rule.anomaly_type,
                    severity=rule.severity,
                    timestamp=timestamp,
                    description=f"Rule '{rule.name}': {rule.description}",
                    detector_name=self.name,
                    metrics=metrics.copy(),
                    metadata={
                        "rule_name": rule.name,
                        "logic": rule.logic,
                        "occurrences": len(rule._occurrences),
                    }
                ))
        
        return anomalies
    
    def reset(self) -> None:
        """Reset detector state"""
        for rule in self.rules:
            rule._occurrences.clear()

