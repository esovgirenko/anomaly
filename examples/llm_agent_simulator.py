"""Симулятор LLM-агентов для тестирования системы обнаружения аномалий"""

import asyncio
import random
from datetime import datetime
from anomaly_detection import AnomalyDetectionSystem, LLMAgent


class LLMAgentSimulator:
    """Симулятор LLM-агента с конфигурируемым поведением"""
    
    def __init__(self, agent_id: str, system: AnomalyDetectionSystem, normal_behavior: dict = None):
        self.agent_id = agent_id
        self.system = system
        self.normal_behavior = normal_behavior or {
            "input_tokens": 1500,
            "output_tokens": 800,
            "latency_ms": 2500.0,
            "cost_usd": 0.04,
            "quality_score": 0.92,
            "context_window_usage": 0.28,
            "coherence_score": 0.90,
            "relevance_score": 0.91,
        }
        self.running = False
    
    async def generate_metrics(self, inject_anomaly: bool = False, anomaly_type: str = "random") -> dict:
        """Генерация метрик, опционально с аномалиями"""
        metrics = {}
        
        for key, base_value in self.normal_behavior.items():
            if inject_anomaly:
                # Инъекция различных типов аномалий
                if anomaly_type == "tokens" and key in ["input_tokens", "output_tokens"]:
                    metrics[key] = base_value * random.uniform(3, 5)  # Скачок токенов
                elif anomaly_type == "latency" and key == "latency_ms":
                    metrics[key] = base_value * random.uniform(5, 10)  # Высокая задержка
                elif anomaly_type == "cost" and key == "cost_usd":
                    metrics[key] = base_value * random.uniform(10, 20)  # Высокая стоимость
                elif anomaly_type == "quality" and key in ["quality_score", "coherence_score", "relevance_score"]:
                    metrics[key] = base_value * random.uniform(0.4, 0.6)  # Низкое качество
                elif anomaly_type == "hallucination":
                    if key == "hallucination_detected":
                        metrics[key] = True
                    elif key == "quality_score":
                        metrics[key] = base_value * 0.7
                else:
                    # Нормальная вариация
                    variation = random.uniform(-0.1, 0.1)
                    metrics[key] = base_value * (1 + variation)
            else:
                # Нормальная вариация
                variation = random.uniform(-0.1, 0.1)
                metrics[key] = base_value * (1 + variation)
        
        # Добавление обязательных полей
        if "total_tokens" not in metrics:
            metrics["total_tokens"] = metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)
        
        if "hallucination_detected" not in metrics:
            metrics["hallucination_detected"] = False
        
        if "factual_errors" not in metrics:
            metrics["factual_errors"] = 0
        
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["agent_id"] = self.agent_id
        metrics["model"] = "gpt-4"
        metrics["provider"] = "openai"
        
        return metrics
    
    async def run(self, duration_seconds: int = 60, anomaly_probability: float = 0.15):
        """Запуск симулятора на указанное время"""
        self.running = True
        monitor = self.system._monitors.get(self.agent_id)
        
        if not monitor:
            print(f"Предупреждение: Монитор не найден для агента {self.agent_id}")
            return
        
        start_time = asyncio.get_event_loop().time()
        iteration = 0
        anomaly_types = ["tokens", "latency", "cost", "quality", "hallucination"]
        
        while self.running:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > duration_seconds:
                break
            
            # Случайная инъекция аномалий
            inject_anomaly = random.random() < anomaly_probability
            anomaly_type = random.choice(anomaly_types) if inject_anomaly else "random"
            
            # Генерация метрик
            metrics = await self.generate_metrics(inject_anomaly=inject_anomaly, anomaly_type=anomaly_type)
            
            # Детекция аномалий
            anomalies = await monitor.detect_anomalies(metrics)
            
            # Регистрация и алерты
            for anomaly in anomalies:
                if self.system.registry.register(anomaly):
                    await self.system.alert_manager.send_alert(anomaly)
                    print(f"[{self.agent_id}] Обнаружена аномалия: {anomaly.description[:80]}...")
            
            iteration += 1
            if iteration % 10 == 0:
                print(f"[{self.agent_id}] Итерация {iteration}, метрики сгенерированы")
            
            await asyncio.sleep(2)  # Генерация метрик каждые 2 секунды
    
    def stop(self):
        """Остановка симулятора"""
        self.running = False


async def run_llm_simulation():
    """Запуск симуляции с несколькими LLM-агентами"""
    
    # Создание системы с LLM-конфигурацией
    system = AnomalyDetectionSystem(config={
        "collection_interval_seconds": 2,
        "detectors": {
            "statistical": {"enabled": True},
            "rule_based": {"enabled": True},
            "llm": {
                "token_usage": {"enabled": True, "max_tokens_threshold": 50000},
                "latency": {"enabled": True, "latency_threshold_ms": 30000},
                "cost": {"enabled": True, "daily_budget_usd": 100.0},
                "quality": {"enabled": True},
                "rate_limit": {"enabled": True},
                "context_overflow": {"enabled": True},
            }
        }
    })
    
    # Регистрация LLM-агентов
    agents = []
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    
    for i, model in enumerate(models):
        agent_id = f"llm_agent_{model.replace('-', '_')}"
        system.register_agent(
            agent_id=agent_id,
            metadata={
                "model": model,
                "provider": "openai" if "gpt" in model else "anthropic",
                "context_window_size": 8192 if "gpt-4" in model else 4096,
                "rate_limit": 500,
            }
        )
        agents.append(agent_id)
    
    print(f"Зарегистрировано {len(agents)} LLM-агентов")
    
    # Запуск мониторинга
    await system.start_monitoring()
    
    # Запуск симуляторов
    simulators = []
    for agent_id in agents:
        simulator = LLMAgentSimulator(agent_id, system)
        simulators.append(simulator)
        # Запуск симулятора в фоне
        asyncio.create_task(simulator.run(duration_seconds=60, anomaly_probability=0.15))
    
    print("Симуляция запущена на 60 секунд...")
    print("Нажмите Ctrl+C для досрочной остановки")
    
    try:
        await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("\nОстановка симуляции...")
    
    # Остановка симуляторов
    for simulator in simulators:
        simulator.stop()
    
    # Остановка мониторинга
    await system.stop_monitoring()
    
    # Вывод статистики
    print("\n" + "="*60)
    print("Результаты симуляции:")
    print("="*60)
    
    overall_stats = system.get_overall_stats(hours=1)
    print(f"\nОбщая статистика:")
    print(f"  Всего аномалий: {overall_stats['total_anomalies']}")
    print(f"  Уникальных агентов: {overall_stats['unique_agents']}")
    print(f"  По типам: {overall_stats['by_type']}")
    print(f"  По серьезности: {overall_stats['by_severity']}")
    
    for agent_id in agents:
        stats = system.get_agent_stats(agent_id, hours=1)
        print(f"\nАгент {agent_id}:")
        print(f"  Всего аномалий: {stats['total_anomalies']}")
        print(f"  Средняя серьезность: {stats['average_severity']:.2f}")
        print(f"  По типам: {stats['by_type']}")
        
        # Показать последние аномалии
        recent_anomalies = system.get_recent_anomalies(agent_id=agent_id, hours=1, limit=5)
        if recent_anomalies:
            print(f"  Последние аномалии:")
            for anomaly in recent_anomalies[:3]:
                print(f"    - [{anomaly.anomaly_type.value}] {anomaly.description[:60]}...")


if __name__ == "__main__":
    asyncio.run(run_llm_simulation())

