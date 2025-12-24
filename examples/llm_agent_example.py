"""Пример использования системы обнаружения аномалий для LLM-агентов"""

import asyncio
from datetime import datetime
from anomaly_detection import AnomalyDetectionSystem, LLMAgent, LLMAgentMetrics


async def example_llm_agent_usage():
    """Пример работы с LLM-агентами"""
    
    # Создание системы с конфигурацией для LLM-агентов
    system = AnomalyDetectionSystem(config={
        "collection_interval_seconds": 5,
        "detectors": {
            "statistical": {"enabled": True},
            "rule_based": {"enabled": True},
            "llm": {
                "token_usage": {"enabled": True, "max_tokens_threshold": 50000},
                "latency": {"enabled": True, "latency_threshold_ms": 30000},
                "cost": {"enabled": True, "daily_budget_usd": 100.0},
                "quality": {"enabled": True, "quality_threshold": 0.7},
                "rate_limit": {"enabled": True},
                "context_overflow": {"enabled": True},
            }
        }
    })
    
    # Регистрация LLM-агента
    llm_agent = system.register_agent(
        agent_id="gpt4_chatbot",
        metrics_endpoint=None,  # Можно указать endpoint для получения метрик
        metadata={
            "model": "gpt-4",
            "provider": "openai",
            "context_window_size": 8192,
            "rate_limit": 500,  # requests per minute
        }
    )
    
    print(f"Зарегистрирован LLM-агент: {llm_agent.agent_id}")
    
    # Запуск мониторинга
    await system.start_monitoring()
    print("Мониторинг запущен...")
    
    # Симуляция метрик от LLM-агента
    monitor = system._monitors.get("gpt4_chatbot")
    if monitor:
        # Нормальные метрики
        normal_metrics = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "latency_ms": 2000.0,
            "cost_usd": 0.03,
            "quality_score": 0.9,
            "context_window_usage": 0.2,
            "model": "gpt-4",
            "provider": "openai",
            "timestamp": datetime.now().isoformat(),
        }
        
        print("\nТестирование с нормальными метриками...")
        anomalies = await monitor.detect_anomalies(normal_metrics)
        print(f"Найдено аномалий: {len(anomalies)}")
        
        # Аномальные метрики - слишком много токенов
        high_token_metrics = {
            "input_tokens": 40000,
            "output_tokens": 20000,
            "total_tokens": 60000,  # Превышает порог
            "latency_ms": 15000.0,
            "cost_usd": 1.2,
            "quality_score": 0.85,
            "context_window_usage": 0.73,
            "model": "gpt-4",
            "provider": "openai",
            "timestamp": datetime.now().isoformat(),
        }
        
        print("\nТестирование с аномальными метриками (высокое использование токенов)...")
        anomalies = await monitor.detect_anomalies(high_token_metrics)
        
        for anomaly in anomalies:
            if system.registry.register(anomaly):
                print(f"\nОбнаружена аномалия:")
                print(f"  Тип: {anomaly.anomaly_type.value}")
                print(f"  Серьезность: {anomaly.severity:.2f}")
                print(f"  Описание: {anomaly.description}")
        
        # Аномальные метрики - низкое качество
        low_quality_metrics = {
            "input_tokens": 2000,
            "output_tokens": 800,
            "total_tokens": 2800,
            "latency_ms": 3000.0,
            "cost_usd": 0.05,
            "quality_score": 0.5,  # Низкое качество
            "hallucination_detected": True,  # Обнаружена галлюцинация
            "factual_errors": 2,
            "coherence_score": 0.6,
            "relevance_score": 0.65,
            "context_window_usage": 0.34,
            "model": "gpt-4",
            "provider": "openai",
            "timestamp": datetime.now().isoformat(),
        }
        
        print("\nТестирование с аномальными метриками (низкое качество)...")
        anomalies = await monitor.detect_anomalies(low_quality_metrics)
        
        for anomaly in anomalies:
            if system.registry.register(anomaly):
                print(f"\nОбнаружена аномалия:")
                print(f"  Тип: {anomaly.anomaly_type.value}")
                print(f"  Серьезность: {anomaly.severity:.2f}")
                print(f"  Описание: {anomaly.description}")
    
    # Получение статистики
    print("\n" + "="*60)
    print("Статистика по агенту:")
    stats = system.get_agent_stats("gpt4_chatbot", hours=1)
    print(f"  Всего аномалий: {stats['total_anomalies']}")
    print(f"  Средняя серьезность: {stats['average_severity']:.2f}")
    print(f"  По типам: {stats['by_type']}")
    
    # Остановка мониторинга
    await system.stop_monitoring()
    print("\nМониторинг остановлен.")


if __name__ == "__main__":
    asyncio.run(example_llm_agent_usage())

