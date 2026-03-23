# Data Pipeline Tools — Spark, Kafka, Flink

## Tổng quan pipeline

```
Raw logs → Kafka → Flink (real-time) → Feature Store → Model
                 → Spark  (batch)    → Training data → Model
```

## Apache Kafka — Message Queue

**Dùng để làm gì:** Thu thập events real-time (clicks, views, purchases) từ nhiều service.

```python
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer: gửi event khi user click
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

event = {
    'user_id': 'u_123',
    'item_id': 'i_456',
    'action': 'click',
    'timestamp': 1704067200,
    'context': {'device': 'mobile', 'page': 'home'}
}
producer.send('user-events', value=event)

# Consumer: xử lý events
consumer = KafkaConsumer(
    'user-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
for message in consumer:
    process_event(message.value)
```

**Đặc điểm:**
- Throughput rất cao (millions events/sec)
- Retention: lưu log N ngày để replay
- Exactly-once delivery với transactions

---

## Apache Spark — Batch Processing

**Dùng để làm gì:** Xử lý data lớn để tạo training data, tính aggregation features, train collaborative filtering.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("RecSys").getOrCreate()

# Đọc interaction logs
interactions = spark.read.parquet("s3://data/interactions/")

# Tính user-item interaction count
user_item_counts = interactions.groupBy("user_id", "item_id") \
    .agg(F.count("*").alias("interaction_count"))

# Tính popularity của item
item_popularity = interactions.groupBy("item_id") \
    .agg(F.count("*").alias("popularity")) \
    .orderBy(F.desc("popularity"))

# Train ALS với Spark MLlib
from pyspark.ml.recommendation import ALS

als = ALS(
    rank=64,
    maxIter=20,
    regParam=0.1,
    userCol="user_id",
    itemCol="item_id",
    ratingCol="interaction_count",
    implicitPrefs=True  # implicit feedback
)
model = als.fit(user_item_counts)

# Generate recommendations cho tất cả users
recommendations = model.recommendForAllUsers(100)
recommendations.write.parquet("s3://data/candidates/als/")
```

**Khi nào dùng Spark:**
- Daily/hourly batch jobs
- Training data preparation
- Offline feature computation
- Model training trên cluster

---

## Apache Flink — Stream Processing

**Dùng để làm gì:** Xử lý events real-time, cập nhật features liên tục, detect patterns trong stream.

```python
# Flink Python API (PyFlink)
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Đọc từ Kafka
t_env.execute_sql("""
    CREATE TABLE user_events (
        user_id STRING,
        item_id STRING,
        action  STRING,
        ts      TIMESTAMP(3),
        WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic'     = 'user-events',
        'format'    = 'json'
    )
""")

# Tính CTR trong cửa sổ 1 giờ (sliding window)
t_env.execute_sql("""
    SELECT
        user_id,
        item_id,
        COUNT(*) FILTER (WHERE action = 'click') AS clicks,
        COUNT(*) AS impressions,
        COUNT(*) FILTER (WHERE action = 'click') * 1.0 / COUNT(*) AS ctr,
        TUMBLE_END(ts, INTERVAL '1' HOUR) AS window_end
    FROM user_events
    GROUP BY
        user_id, item_id,
        TUMBLE(ts, INTERVAL '1' HOUR)
""")
```

**Khi nào dùng Flink vs Spark:**

| | Spark Streaming | Flink |
|---|---|---|
| Latency | Mini-batch (~seconds) | True streaming (ms) |
| State management | Có nhưng phức tạp | Native, rất mạnh |
| Exactly-once | Có | Có |
| Dùng khi | Gần real-time OK | Cần real-time thật sự |

---

## Kiến trúc Pipeline thực tế

```
User actions
     │
     ▼
  Kafka (raw events)
     │
     ├──► Flink ──► Redis (online features, TTL=1h)
     │              - user recent clicks
     │              - session features
     │
     └──► Spark (daily batch)
              │
              ├──► Parquet/Hive (training data)
              ├──► ALS/MF model training
              └──► Feature Store offline (Feast)
                        │
                        └──► Feature Store online (Redis)
```

## Công cụ bổ sung

- **Airflow**: orchestrate batch jobs (DAG scheduling)
- **dbt**: transform data trong warehouse
- **Great Expectations**: data quality validation
- **MLflow**: track experiments, model versioning
