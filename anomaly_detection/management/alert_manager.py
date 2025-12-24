"""Alert manager for sending notifications about anomalies"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json

from anomaly_detection.core.anomaly import Anomaly, Severity


class AlertChannel(Enum):
    """Types of alert channels"""
    CONSOLE = "console"
    SLACK = "slack"
    EMAIL = "email"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"


class AlertTemplate:
    """Template for alert messages"""
    
    def __init__(self, template: str, format_type: str = "text"):
        self.template = template
        self.format_type = format_type  # "text", "json", "html"
    
    def render(self, anomaly: Anomaly, context: Optional[Dict] = None) -> str:
        """Render template with anomaly data"""
        context = context or {}
        
        data = {
            "agent_id": anomaly.agent_id,
            "anomaly_type": anomaly.anomaly_type.value,
            "severity": anomaly.severity,
            "severity_label": self._get_severity_label(anomaly.severity),
            "timestamp": anomaly.timestamp.isoformat(),
            "description": anomaly.description,
            "detector_name": anomaly.detector_name,
            "metrics": json.dumps(anomaly.metrics, indent=2) if self.format_type == "text" else anomaly.metrics,
            **context
        }
        
        if self.format_type == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            # Simple template substitution
            message = self.template
            for key, value in data.items():
                message = message.replace(f"{{{key}}}", str(value))
            return message
    
    @staticmethod
    def _get_severity_label(severity: float) -> str:
        """Get human-readable severity label"""
        if severity < 0.4:
            return "LOW"
        elif severity < 0.6:
            return "MEDIUM"
        elif severity < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"


class AlertRule:
    """Rule for when to send alerts"""
    
    def __init__(
        self,
        min_severity: float = 0.5,
        channels: Optional[List[AlertChannel]] = None,
        cooldown: Optional[timedelta] = None,
        template: Optional[AlertTemplate] = None
    ):
        self.min_severity = min_severity
        self.channels = channels or [AlertChannel.CONSOLE]
        self.cooldown = cooldown
        self.template = template
        self._last_alert: Dict[str, datetime] = {}  # per agent/channel
    
    def should_alert(self, anomaly: Anomaly, channel: AlertChannel) -> bool:
        """Check if alert should be sent"""
        # Check severity
        if anomaly.severity < self.min_severity:
            return False
        
        # Check channel is enabled
        if channel not in self.channels:
            return False
        
        # Check cooldown
        if self.cooldown:
            key = f"{anomaly.agent_id}:{channel.value}"
            last_alert = self._last_alert.get(key)
            if last_alert and datetime.now() - last_alert < self.cooldown:
                return False
        
        return True
    
    def record_alert(self, anomaly: Anomaly, channel: AlertChannel) -> None:
        """Record that an alert was sent"""
        key = f"{anomaly.agent_id}:{channel.value}"
        self._last_alert[key] = datetime.now()


class AlertManager:
    """
    Manages alerting for detected anomalies
    
    Features:
    - Multiple alert channels (console, Slack, Email, etc.)
    - Configurable alert rules
    - Message templates
    - Cooldown periods to prevent spam
    - Escalation support
    """
    
    def __init__(
        self,
        default_rule: Optional[AlertRule] = None,
        custom_handlers: Optional[Dict[AlertChannel, Callable]] = None
    ):
        """
        Initialize alert manager
        
        Args:
            default_rule: Default alert rule to use
            custom_handlers: Custom handlers for alert channels
        """
        self.default_rule = default_rule or AlertRule(
            min_severity=0.5,
            channels=[AlertChannel.CONSOLE],
            template=AlertTemplate(
                "[{severity_label}] Anomaly detected in {agent_id}\n"
                "Type: {anomaly_type}\n"
                "Time: {timestamp}\n"
                "Description: {description}\n"
                "Detector: {detector_name}"
            )
        )
        self.rules: List[AlertRule] = [self.default_rule]
        self.custom_handlers = custom_handlers or {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default handlers for alert channels"""
        if AlertChannel.CONSOLE not in self.custom_handlers:
            self.custom_handlers[AlertChannel.CONSOLE] = self._console_handler
        
        # Add other default handlers as needed
        # For Slack, Email, etc., users should provide their own handlers
    
    async def send_alert(self, anomaly: Anomaly, rule: Optional[AlertRule] = None) -> None:
        """
        Send alert for an anomaly
        
        Args:
            anomaly: The anomaly to alert about
            rule: Optional alert rule to use (defaults to default_rule)
        """
        rule = rule or self.default_rule
        
        # Try all alert rules
        for alert_rule in self.rules:
            if anomaly.severity < alert_rule.min_severity:
                continue
            
            for channel in alert_rule.channels:
                if not alert_rule.should_alert(anomaly, channel):
                    continue
                
                try:
                    # Get handler
                    handler = self.custom_handlers.get(channel)
                    if not handler:
                        print(f"No handler configured for channel {channel.value}")
                        continue
                    
                    # Render message
                    template = alert_rule.template or self.default_rule.template
                    message = template.render(anomaly)
                    
                    # Send alert
                    if asyncio.iscoroutinefunction(handler):
                        await handler(anomaly, message)
                    else:
                        handler(anomaly, message)
                    
                    # Record alert
                    alert_rule.record_alert(anomaly, channel)
                    
                except Exception as e:
                    print(f"Error sending alert via {channel.value}: {e}")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule"""
        self.rules.append(rule)
    
    def register_handler(self, channel: AlertChannel, handler: Callable) -> None:
        """Register a custom handler for an alert channel"""
        self.custom_handlers[channel] = handler
    
    @staticmethod
    def _console_handler(anomaly: Anomaly, message: str) -> None:
        """Default console handler"""
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60 + "\n")
    
    @staticmethod
    async def slack_handler_factory(webhook_url: str):
        """Factory for creating Slack webhook handler"""
        import aiohttp
        
        async def handler(anomaly: Anomaly, message: str) -> None:
            async with aiohttp.ClientSession() as session:
                payload = {"text": message}
                async with session.post(webhook_url, json=payload) as resp:
                    if resp.status != 200:
                        raise Exception(f"Slack webhook failed: {resp.status}")
        
        return handler
    
    @staticmethod
    async def webhook_handler_factory(url: str, headers: Optional[Dict] = None):
        """Factory for creating generic webhook handler"""
        import aiohttp
        
        async def handler(anomaly: Anomaly, message: str) -> None:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "anomaly": anomaly.to_dict(),
                    "message": message
                }
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status not in (200, 201, 204):
                        raise Exception(f"Webhook failed: {resp.status}")
        
        return handler

