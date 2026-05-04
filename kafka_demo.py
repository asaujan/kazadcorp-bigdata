"""
KazAdCorp Kafka Demo — Потоковая обработка рекламных событий
Симулирует реальный Kafka стенд без Docker
Показывает: Producer → Topic → Consumer → Real-time аналитика
"""
import json
import time
import random
import threading
import queue
from datetime import datetime
from collections import defaultdict
import pandas as pd

print("="*65)
print("KazAdCorp Kafka Demo")
print("Потоковая обработка рекламных событий в реальном времени")
print("="*65)

# ── КОНФИГУРАЦИЯ ──────────────────────────────────────────────
KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",  # реальный адрес в продакшене
    "topic": "kazadcorp_ad_events",
    "partitions": 3,
    "replication_factor": 1,
    "retention_ms": 86400000,  # 24 часа
}

ADVERTISERS = ["Kaspi", "Halyk Bank", "Air Astana", "Wildberries",
               "Kolesa.kz", "Kcell", "Forte Bank", "OLX"]
PLATFORMS   = ["Facebook", "Instagram", "Facebook,Instagram"]
AGE_GROUPS  = ["18-24", "25-34", "35-44", "45-54", "55+"]
EVENT_TYPES = ["impression", "click", "conversion", "bounce"]

# Весовые коэффициенты событий (реалистичная воронка)
EVENT_WEIGHTS = {
    "impression":  0.940,
    "click":       0.052,
    "conversion":  0.006,
    "bounce":      0.002,
}

# ── KAFKA TOPIC (in-memory симуляция) ────────────────────────
class KafkaTopic:
    """In-memory симуляция Kafka топика с партициями"""
    def __init__(self, name, partitions=3):
        self.name = name
        self.partitions = [queue.Queue() for _ in range(partitions)]
        self.offsets = [0] * partitions
        self.total_messages = 0

    def send(self, message, key=None):
        """Отправка сообщения в партицию (round-robin или по ключу)"""
        if key:
            partition_id = hash(key) % len(self.partitions)
        else:
            partition_id = self.total_messages % len(self.partitions)
        
        msg = {
            "partition": partition_id,
            "offset": self.offsets[partition_id],
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "value": message
        }
        self.partitions[partition_id].put(msg)
        self.offsets[partition_id] += 1
        self.total_messages += 1
        return partition_id

    def poll(self, partition_id, timeout=0.1):
        """Чтение сообщений из партиции"""
        messages = []
        try:
            while True:
                msg = self.partitions[partition_id].get(timeout=timeout)
                messages.append(msg)
        except queue.Empty:
            pass
        return messages

    def stats(self):
        return {
            "topic": self.name,
            "partitions": len(self.partitions),
            "total_messages": self.total_messages,
            "offsets": self.offsets,
        }

# ── PRODUCER ──────────────────────────────────────────────────
class AdEventProducer:
    """
    Симулирует Kafka Producer который отправляет
    рекламные события в реальном времени
    """
    def __init__(self, topic: KafkaTopic):
        self.topic = topic
        self.produced = 0
        self.running = False

    def generate_event(self):
        """Генерирует реалистичное рекламное событие"""
        advertiser = random.choice(ADVERTISERS)
        event_type = random.choices(
            list(EVENT_WEIGHTS.keys()),
            weights=list(EVENT_WEIGHTS.values())
        )[0]

        # Реалистичные метрики по типу события
        if event_type == "impression":
            revenue = 0
            cost = round(random.uniform(0.5, 5.0), 2)
        elif event_type == "click":
            revenue = 0
            cost = round(random.uniform(50, 300), 2)
        elif event_type == "conversion":
            revenue = round(random.uniform(5000, 80000), 2)
            cost = round(random.uniform(200, 800), 2)
        else:  # bounce
            revenue = 0
            cost = round(random.uniform(30, 150), 2)

        return {
            "event_id":   f"evt_{int(time.time()*1000)}_{random.randint(1000,9999)}",
            "event_type": event_type,
            "advertiser": advertiser,
            "platform":   random.choice(PLATFORMS),
            "age_group":  random.choice(AGE_GROUPS),
            "gender":     random.choice(["M", "F"]),
            "revenue_kzt": revenue,
            "cost_kzt":    cost,
            "campaign_id": f"camp_{advertiser[:3].upper()}_{random.randint(100,999)}",
            "timestamp":   datetime.now().isoformat(),
            "session_id":  f"sess_{random.randint(10000, 99999)}",
        }

    def produce(self, n_events=100, delay=0.01):
        """Отправляет N событий в топик"""
        self.running = True
        for i in range(n_events):
            if not self.running:
                break
            event = self.generate_event()
            # Ключ = advertiser для гарантии порядка по рекламодателю
            partition = self.topic.send(
                message=event,
                key=event["advertiser"]
            )
            self.produced += 1
            time.sleep(delay)
        return self.produced

# ── CONSUMER ──────────────────────────────────────────────────
class AdEventConsumer:
    """
    Симулирует Kafka Consumer Group
    Обрабатывает события и считает real-time метрики
    """
    def __init__(self, topic: KafkaTopic, consumer_id: str, partitions: list):
        self.topic = topic
        self.consumer_id = consumer_id
        self.partitions = partitions  # партиции которые слушает этот consumer
        self.consumed = 0
        self.metrics = defaultdict(lambda: {
            "impressions": 0, "clicks": 0,
            "conversions": 0, "bounces": 0,
            "revenue": 0.0, "cost": 0.0
        })
        self.running = False

    def process_event(self, event):
        """Обрабатывает одно событие и обновляет метрики"""
        adv = event["advertiser"]
        etype = event["event_type"]

        if etype == "impression":
            self.metrics[adv]["impressions"] += 1
        elif etype == "click":
            self.metrics[adv]["clicks"] += 1
        elif etype == "conversion":
            self.metrics[adv]["conversions"] += 1
            self.metrics[adv]["revenue"] += event["revenue_kzt"]
        elif etype == "bounce":
            self.metrics[adv]["bounces"] += 1

        self.metrics[adv]["cost"] += event["cost_kzt"]
        self.consumed += 1

    def consume(self, max_messages=1000):
        """Читает сообщения из назначенных партиций"""
        self.running = True
        consumed = 0
        while self.running and consumed < max_messages:
            for partition_id in self.partitions:
                messages = self.topic.poll(partition_id, timeout=0.05)
                for msg in messages:
                    self.process_event(msg["value"])
                    consumed += 1
            time.sleep(0.01)
        return consumed

    def get_realtime_metrics(self):
        """Возвращает real-time метрики"""
        rows = []
        for adv, m in self.metrics.items():
            impressions = m["impressions"]
            clicks = m["clicks"]
            conversions = m["conversions"]
            cost = m["cost"]
            revenue = m["revenue"]

            ctr  = round(clicks/impressions*100, 3) if impressions > 0 else 0
            cr   = round(conversions/clicks*100, 3) if clicks > 0 else 0
            roas = round(revenue/cost, 2) if cost > 0 else 0
            cac  = round(cost/conversions, 0) if conversions > 0 else None

            rows.append({
                "advertiser":   adv,
                "impressions":  impressions,
                "clicks":       clicks,
                "conversions":  conversions,
                "ctr_pct":      ctr,
                "cr_pct":       cr,
                "cost_kzt":     round(cost, 0),
                "revenue_kzt":  round(revenue, 0),
                "roas":         roas,
                "cac_kzt":      cac,
            })
        return pd.DataFrame(rows).sort_values("roas", ascending=False)

# ── STREAM PROCESSOR ──────────────────────────────────────────
class StreamProcessor:
    """
    Симулирует Apache Flink / Spark Streaming processor
    Обрабатывает события в скользящих временных окнах
    """
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.window = []
        self.alerts = []
        self.processed_windows = 0

    def process_window(self, events):
        """Обрабатывает одно временное окно событий"""
        if not events:
            return

        self.window.extend(events)
        self.processed_windows += 1

        # Считаем метрики по окну
        total = len(events)
        clicks = sum(1 for e in events if e["event_type"] == "click")
        convs  = sum(1 for e in events if e["event_type"] == "conversion")
        ctr    = clicks / total * 100 if total > 0 else 0

        # Алерты в реальном времени
        if ctr < 0.5:
            self.alerts.append({
                "window": self.processed_windows,
                "type": "LOW_CTR",
                "message": f"CTR = {ctr:.2f}% — ниже порога 0.5%",
                "severity": "WARNING"
            })
        if convs == 0 and total > 30:
            self.alerts.append({
                "window": self.processed_windows,
                "type": "ZERO_CONVERSIONS",
                "message": f"Нет конверсий в окне {total} событий",
                "severity": "CRITICAL"
            })

    def get_alerts(self):
        return self.alerts[-5:]  # последние 5 алертов

# ── MAIN DEMO ─────────────────────────────────────────────────
def main():
    print("\n⚙️  Конфигурация Kafka:")
    print(f"   Bootstrap servers: {KAFKA_CONFIG['bootstrap_servers']}")
    print(f"   Topic: {KAFKA_CONFIG['topic']}")
    print(f"   Партиций: {KAFKA_CONFIG['partitions']}")
    print(f"   Retention: 24 часа")

    # Создаём топик
    topic = KafkaTopic(
        name=KAFKA_CONFIG["topic"],
        partitions=KAFKA_CONFIG["partitions"]
    )

    # Создаём producer и consumers (Consumer Group)
    producer = AdEventProducer(topic)
    consumer1 = AdEventConsumer(topic, "consumer-1", partitions=[0])
    consumer2 = AdEventConsumer(topic, "consumer-2", partitions=[1])
    consumer3 = AdEventConsumer(topic, "consumer-3", partitions=[2])
    processor = StreamProcessor(window_size=50)

    print("\n" + "─"*65)
    print("ЭТАП 1: Генерация потока рекламных событий (Producer)")
    print("─"*65)
    print("Запуск Producer — отправка событий в топик kazadcorp_ad_events...")

    N_EVENTS = 2000

    # Producer в отдельном потоке
    producer_thread = threading.Thread(
        target=producer.produce,
        kwargs={"n_events": N_EVENTS, "delay": 0.001}
    )

    # Consumers в параллельных потоках
    consumer_threads = [
        threading.Thread(target=c.consume, kwargs={"max_messages": N_EVENTS})
        for c in [consumer1, consumer2, consumer3]
    ]

    t_start = time.time()
    producer_thread.start()
    for t in consumer_threads:
        t.start()

    # Мониторинг в реальном времени
    print(f"\n{'Время':>8} {'Отправлено':>12} {'Обработано':>12} {'Скорость':>12}")
    print("─" * 48)

    while producer_thread.is_alive():
        elapsed = time.time() - t_start
        produced = producer.produced
        consumed = consumer1.consumed + consumer2.consumed + consumer3.consumed
        speed = produced / elapsed if elapsed > 0 else 0
        print(f"{elapsed:>7.1f}s {produced:>12,} {consumed:>12,} {speed:>10.0f}/сек")
        time.sleep(0.5)

    producer_thread.join()
    for t in consumer_threads:
        t.join(timeout=2)

    elapsed = time.time() - t_start
    total_consumed = consumer1.consumed + consumer2.consumed + consumer3.consumed

    print(f"\n✅ Producer завершил отправку {N_EVENTS:,} событий за {elapsed:.2f} сек")
    print(f"✅ Throughput: {N_EVENTS/elapsed:.0f} событий/сек")
    print(f"✅ Всего обработано consumers: {total_consumed:,}")

    # Статистика топика
    stats = topic.stats()
    print(f"\n📊 Статистика топика '{stats['topic']}':")
    for i, offset in enumerate(stats["offsets"]):
        print(f"   Партиция {i}: {offset:,} сообщений")

    print("\n" + "─"*65)
    print("ЭТАП 2: Real-time аналитика (Consumer Group)")
    print("─"*65)

    # Объединяем метрики всех consumers
    all_metrics = defaultdict(lambda: {
        "impressions": 0, "clicks": 0, "conversions": 0,
        "bounces": 0, "revenue": 0.0, "cost": 0.0
    })

    for consumer in [consumer1, consumer2, consumer3]:
        for adv, m in consumer.metrics.items():
            for key, val in m.items():
                all_metrics[adv][key] += val

    rows = []
    for adv, m in all_metrics.items():
        imp  = m["impressions"]
        clk  = m["clicks"]
        conv = m["conversions"]
        cost = m["cost"]
        rev  = m["revenue"]
        ctr  = round(clk/imp*100, 3) if imp > 0 else 0
        cr   = round(conv/clk*100, 3) if clk > 0 else 0
        roas = round(rev/cost, 2) if cost > 0 else 0
        cac  = round(cost/conv, 0) if conv > 0 else 0
        rows.append({
            "Рекламодатель": adv,
            "Показы": imp,
            "Клики": clk,
            "CTR%": ctr,
            "Конверсии": conv,
            "CR%": cr,
            "Затраты(тг)": round(cost, 0),
            "Выручка(тг)": round(rev, 0),
            "ROAS": roas,
            "CAC(тг)": cac,
        })

    df = pd.DataFrame(rows).sort_values("ROAS", ascending=False)
    print("\n📊 Real-time метрики по рекламодателям (обработано из потока):")
    print(df.to_string(index=False))

    print("\n" + "─"*65)
    print("ЭТАП 3: Stream Processing — временные окна и алерты")
    print("─"*65)

    # Симулируем обработку в скользящих окнах
    print("Обработка событий в окнах по 50 событий (Flink/Spark Streaming)...")
    all_events = []
    for consumer in [consumer1, consumer2, consumer3]:
        # Восстанавливаем события из метрик для демонстрации
        for adv, m in consumer.metrics.items():
            for _ in range(m["impressions"]):
                all_events.append({"event_type":"impression","advertiser":adv,"cost_kzt":1.0})
            for _ in range(m["clicks"]):
                all_events.append({"event_type":"click","advertiser":adv,"cost_kzt":100.0})
            for _ in range(m["conversions"]):
                all_events.append({"event_type":"conversion","advertiser":adv,"cost_kzt":500.0})

    random.shuffle(all_events)
    window_size = 100
    for i in range(0, min(len(all_events), 1000), window_size):
        window_events = all_events[i:i+window_size]
        processor.process_window(window_events)

    print(f"✅ Обработано временных окон: {processor.processed_windows}")

    alerts = processor.get_alerts()
    if alerts:
        print(f"\n🚨 Real-time алерты ({len(alerts)} последних):")
        for alert in alerts:
            icon = "🔴" if alert["severity"]=="CRITICAL" else "🟡"
            print(f"   {icon} Окно #{alert['window']}: [{alert['type']}] {alert['message']}")
    else:
        print("\n✅ Алертов нет — показатели в норме")

    print("\n" + "─"*65)
    print("ЭТАП 4: Архитектура Lambda — итоговая схема")
    print("─"*65)
    print("""
  [Facebook Ads API]  [Google Ads API]  [Kaspi Pay API]
         │                   │                 │
         └──────────┬─────────┘                │
                    ▼                          │
         ┌──────────────────────┐              │
         │  Apache Kafka        │◄─────────────┘
         │  Topic: ad_events    │
         │  Partitions: 3       │
         │  50,000 events/sec   │
         └──────┬───────────────┘
                │
     ┌──────────┴──────────┐
     ▼                     ▼
┌─────────────┐    ┌──────────────────┐
│ BATCH LAYER │    │  SPEED LAYER     │
│ Apache Spark│    │  Apache Flink    │
│ (исторические│   │  (30 сек задержка│
│  данные)    │    │  real-time KPI)  │
└──────┬──────┘    └────────┬─────────┘
       │                    │
       └──────────┬──────────┘
                  ▼
         ┌─────────────────┐
         │  SERVING LAYER  │
         │  ClickHouse     │
         │  + Apache       │
         │    Superset     │
         │  SLA: 200 мс    │
         └─────────────────┘
    """)

    # Финальный отчёт
    print("="*65)
    print("ИТОГОВЫЙ ОТЧЁТ KAFKA DEMO")
    print("="*65)
    print(f"Событий сгенерировано:  {N_EVENTS:,}")
    print(f"Throughput:             {N_EVENTS/elapsed:.0f} событий/сек")
    print(f"Consumer Group:         3 consumers, 3 партиции")
    print(f"Временных окон:         {processor.processed_windows}")
    print(f"Real-time алертов:      {len(processor.alerts)}")
    print(f"Уникальных рекламодат.: {len(all_metrics)}")
    print(f"\n✅ Kafka Demo завершён успешно!")
    print(f"   Демонстрирует: Producer → Partitioned Topic → Consumer Group")
    print(f"   → Stream Processing → Real-time KPI → Alerts")

if __name__ == "__main__":
    main()
