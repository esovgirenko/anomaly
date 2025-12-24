# Система обнаружения аномалий для мультиагентных систем

Комплексная система обнаружения аномалий, разработанная для мониторинга и анализа поведения агентов в распределенных мультиагентных системах.

## Основные возможности

- 🔍 **Множественные методы обнаружения**: статистические, ML, правил-основанные и временные ряды
- 📊 **Реал-тайм мониторинг**: непрерывный сбор метрик и обнаружение аномалий
- 🚨 **Система оповещений**: настраиваемые каналы уведомлений
- 📈 **Интеграция с Prometheus**: экспорт метрик для мониторинга
- 🌐 **REST API**: полный API для управления системой
- 📡 **WebSocket**: real-time уведомления об аномалиях
- 🐳 **Docker поддержка**: готовое к развертыванию решение

## Архитектура

### Компоненты

1. **Core Components**:
   - `Anomaly`: Dataclass для представления аномалий
   - `AnomalyDetector`: Абстрактный базовый класс для детекторов
   - `AgentMonitor`: Мониторинг отдельных агентов

2. **Detectors**:
   - `StatisticalDetector`: Z-score, IQR, скользящее среднее
   - `MLDetector`: Isolation Forest, One-Class SVM
   - `RuleBasedDetector`: Конфигурируемые правила
   - `TimeSeriesDetector`: Анализ временных рядов с Prophet

3. **Management**:
   - `AnomalyRegistry`: Регистрация и дедупликация аномалий
   - `AlertManager`: Управление оповещениями
   - `MetricsCollector`: Сбор и хранение метрик

4. **System**:
   - `AnomalyDetectionSystem`: Главный класс системы

## Установка

### Требования

- Python 3.9+
- pip

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Конфигурация

1. Скопируйте пример конфигурации агентов:
```bash
cp config/agents.yaml.example config/agents.yaml
```

2. Отредактируйте `config/config.yaml` и `config/rules.yaml` под ваши нужды

## Использование

### Базовое использование

```python
from anomaly_detection import AnomalyDetectionSystem

# Создание системы
system = AnomalyDetectionSystem(config='config/config.yaml')

# Регистрация агента
agent = system.register_agent(
    agent_id='agent_1',
    metrics_endpoint='http://agent1:9090/metrics',
    metadata={'name': 'Agent 1', 'environment': 'production'}
)

# Запуск мониторинга
await system.start_monitoring()

# Получение аномалий
anomalies = system.get_recent_anomalies(hours=24)
for anomaly in anomalies:
    print(f"Anomaly: {anomaly.description}, Severity: {anomaly.severity}")
```

### Запуск с API сервером

```bash
python main.py --config config/config.yaml --agents-config config/agents.yaml
```

### Запуск без API (только мониторинг)

```bash
python main.py --config config/config.yaml --no-api
```

### Использование Docker Compose

```bash
docker-compose up -d
```

Система будет доступна по адресам:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## API Endpoints

### Агенты

- `POST /agents/register` - Регистрация нового агента
- `GET /agents` - Список всех агентов
- `GET /agents/{agent_id}` - Информация об агенте
- `DELETE /agents/{agent_id}` - Удаление агента
- `POST /agents/{agent_id}/check` - Ручная проверка агента

### Аномалии

- `GET /anomalies` - Получить список аномалий
  - Параметры: `hours`, `agent_id`, `anomaly_type`, `min_severity`, `limit`
- `GET /agents/{agent_id}/stats` - Статистика по агенту
- `GET /stats` - Общая статистика системы

### Мониторинг

- `POST /monitoring/start` - Запустить мониторинг
- `POST /monitoring/stop` - Остановить мониторинг
- `GET /health` - Health check

### Метрики

- `GET /metrics` - Prometheus метрики

### WebSocket

- `WS /ws/anomalies` - Real-time поток аномалий

## Примеры

### Пример 1: Базовое использование

См. `examples/basic_usage.py`

```bash
python examples/basic_usage.py
```

### Пример 2: Симулятор агентов

См. `examples/agent_simulator.py`

```bash
python examples/agent_simulator.py
```

## Конфигурация детекторов

### Statistical Detector

```yaml
detectors:
  statistical:
    enabled: true
    z_score_threshold: 3.0
    iqr_multiplier: 1.5
    window_size: 100
```

### ML Detector

```yaml
detectors:
  ml:
    enabled: true
    method: isolation_forest  # or "one_class_svm"
    contamination: 0.1
    n_estimators: 100
    online_learning: false
```

### Rule-Based Detector

Создайте правила в `config/rules.yaml`:

```yaml
rules:
  - name: high_cpu_usage
    description: CPU usage exceeds 90%
    anomaly_type: performance
    severity: 0.8
    logic: AND
    conditions:
      - metric: cpu_usage
        operator: ">"
        value: 90.0
```

### Time Series Detector

```yaml
detectors:
  timeseries:
    enabled: true
    window_size: 100
    use_prophet: true
    trend_threshold: 2.0
```

## Типы аномалий

- `BEHAVIORAL` - Поведенческие аномалии
- `PERFORMANCE` - Проблемы производительности
- `COMMUNICATION` - Проблемы коммуникации
- `TEMPORAL` - Временные аномалии
- `SEMANTIC` - Семантические аномалии

## Метрики Prometheus

Система экспортирует следующие метрики:

- `anomalies_detected_total` - Общее количество обнаруженных аномалий
- `anomaly_severity` - Распределение severity аномалий
- `agents_monitored` - Количество мониторируемых агентов
- `active_anomalies` - Количество активных аномалий

## Производительность

Система оптимизирована для:
- Низкой задержки обнаружения (< 1 секунда)
- Высокой пропускной способности (10K+ метрик/сек)
- Минимального потребления памяти
- Горизонтального масштабирования

## Расширение системы

### Добавление нового детектора

```python
from anomaly_detection.core.detector import AnomalyDetector
from anomaly_detection.core.anomaly import Anomaly, AnomalyType

class CustomDetector(AnomalyDetector):
    def detect(self, agent_id: str, metrics: dict, historical_data=None):
        # Ваша логика обнаружения
        anomalies = []
        # ...
        return anomalies

# Использование
system.add_detector(CustomDetector(name="custom"))
```

### Кастомный обработчик алертов

```python
from anomaly_detection.management.alert_manager import AlertManager, AlertChannel

async def my_slack_handler(anomaly, message):
    # Ваша логика отправки в Slack
    pass

alert_manager.register_handler(AlertChannel.SLACK, my_slack_handler)
```

## Разработка

### Запуск тестов

```bash
pytest tests/
```

### Форматирование кода

```bash
black anomaly_detection/
```

## Лицензия

MIT

## Авторы

Система разработана для использования в распределенных мультиагентных системах.

## Поддержка

Для вопросов и предложений создайте issue в репозитории проекта.

