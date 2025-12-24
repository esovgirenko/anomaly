"""Agent simulator for testing anomaly detection"""

import asyncio
import random
from datetime import datetime
from anomaly_detection import AnomalyDetectionSystem


class AgentSimulator:
    """Simulates an agent with configurable behavior"""
    
    def __init__(self, agent_id: str, system: AnomalyDetectionSystem, normal_behavior: dict = None):
        self.agent_id = agent_id
        self.system = system
        self.normal_behavior = normal_behavior or {
            "cpu_usage": 50.0,
            "memory_usage": 60.0,
            "response_time_ms": 150.0,
            "requests_per_second": 100.0,
            "error_rate": 0.01,
        }
        self.running = False
    
    async def generate_metrics(self, inject_anomaly: bool = False) -> dict:
        """Generate metrics, optionally injecting anomalies"""
        metrics = {}
        
        for key, base_value in self.normal_behavior.items():
            if inject_anomaly:
                # Inject different types of anomalies
                if key == "cpu_usage":
                    metrics[key] = base_value + random.uniform(40, 60)  # High CPU
                elif key == "memory_usage":
                    metrics[key] = base_value + random.uniform(25, 35)  # High memory
                elif key == "response_time_ms":
                    metrics[key] = base_value * random.uniform(5, 10)  # Slow response
                elif key == "error_rate":
                    metrics[key] = random.uniform(0.1, 0.3)  # High error rate
                else:
                    metrics[key] = base_value + random.uniform(-10, 10)
            else:
                # Normal variation
                variation = random.uniform(-0.1, 0.1)  # ±10% variation
                metrics[key] = base_value * (1 + variation)
        
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["agent_id"] = self.agent_id
        
        return metrics
    
    async def run(self, duration_seconds: int = 60, anomaly_probability: float = 0.1):
        """Run simulator for specified duration"""
        self.running = True
        monitor = self.system._monitors.get(self.agent_id)
        
        if not monitor:
            print(f"Warning: Monitor not found for agent {self.agent_id}")
            return
        
        start_time = asyncio.get_event_loop().time()
        iteration = 0
        
        while self.running:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > duration_seconds:
                break
            
            # Randomly inject anomalies
            inject_anomaly = random.random() < anomaly_probability
            
            # Generate metrics
            metrics = await self.generate_metrics(inject_anomaly=inject_anomaly)
            
            # Detect anomalies
            anomalies = await monitor.detect_anomalies(metrics)
            
            # Register and alert
            for anomaly in anomalies:
                if self.system.registry.register(anomaly):
                    await self.system.alert_manager.send_alert(anomaly)
                    print(f"[{self.agent_id}] Anomaly detected: {anomaly.description}")
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"[{self.agent_id}] Iteration {iteration}, metrics generated")
            
            await asyncio.sleep(2)  # Generate metrics every 2 seconds
    
    def stop(self):
        """Stop the simulator"""
        self.running = False


async def run_simulation():
    """Run a simulation with multiple agents"""
    
    # Create system
    system = AnomalyDetectionSystem(config={
        "collection_interval_seconds": 2,
        "detectors": {
            "statistical": {"enabled": True},
            "rule_based": {"enabled": True},
        },
        "rules": [
            {
                "name": "high_cpu",
                "description": "CPU usage above 80%",
                "anomaly_type": "performance",
                "severity": 0.7,
                "logic": "AND",
                "conditions": [{"metric": "cpu_usage", "operator": ">", "value": 80.0}]
            },
            {
                "name": "high_error_rate",
                "description": "Error rate above 5%",
                "anomaly_type": "communication",
                "severity": 0.8,
                "logic": "AND",
                "conditions": [{"metric": "error_rate", "operator": ">", "value": 0.05}]
            },
        ]
    })
    
    # Register agents
    agents = []
    for i in range(3):
        agent_id = f"simulated_agent_{i+1}"
        system.register_agent(
            agent_id=agent_id,
            metadata={"type": "simulated", "index": i}
        )
        agents.append(agent_id)
    
    print(f"Registered {len(agents)} agents")
    
    # Start monitoring
    await system.start_monitoring()
    
    # Run simulators
    simulators = []
    for agent_id in agents:
        simulator = AgentSimulator(agent_id, system)
        simulators.append(simulator)
        # Start simulator in background
        asyncio.create_task(simulator.run(duration_seconds=60, anomaly_probability=0.15))
    
    print("Simulation running for 60 seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    # Stop simulators
    for simulator in simulators:
        simulator.stop()
    
    # Stop monitoring
    await system.stop_monitoring()
    
    # Print statistics
    print("\n" + "="*60)
    print("Simulation Results:")
    print("="*60)
    
    overall_stats = system.get_overall_stats(hours=1)
    print(f"\nOverall Statistics:")
    print(f"  Total anomalies: {overall_stats['total_anomalies']}")
    print(f"  Unique agents: {overall_stats['unique_agents']}")
    print(f"  By type: {overall_stats['by_type']}")
    print(f"  By severity: {overall_stats['by_severity']}")
    
    for agent_id in agents:
        stats = system.get_agent_stats(agent_id, hours=1)
        print(f"\nAgent {agent_id}:")
        print(f"  Total anomalies: {stats['total_anomalies']}")
        print(f"  Average severity: {stats['average_severity']:.2f}")
        print(f"  By type: {stats['by_type']}")


if __name__ == "__main__":
    asyncio.run(run_simulation())

