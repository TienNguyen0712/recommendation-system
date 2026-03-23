# Feature Engineering cho Recommender System

## Phân loại features

### User features
```
Demographic:    age, gender, location, language
Behavioral:     session length, avg click rate, purchase history
Preference:     genre preference, price range, brand affinity
Temporal:       time of day, day of week, recency of activity
```

### Item features
```
Content:        title, description, category, tags, brand
Metadata:       price, release date, popularity, avg rating
Visual:         thumbnail embedding (CNN), color palette
Textual:        TF-IDF, BERT embedding của description
```

### Context features (rất quan trọng)
```
Device:         mobile vs desktop, app version
Time:           hour, weekday, season
Location:       city, country
Session:        last N items clicked, current search query
```

### Interaction features (cross features)
```
user × item:    user đã xem category này bao nhiêu lần
user × time:    user thường active vào giờ nào
item × time:    item trending vào thời điểm nào
```

## Feature Engineering kỹ thuật

### 1. Encoding categorical features

```python
# Label encoding (ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_enc'] = le.fit_transform(df['category'])

# One-hot (low cardinality)
df = pd.get_dummies(df, columns=['device_type'])

# Target encoding (high cardinality — thận trọng với leakage)
category_mean = df.groupby('category')['label'].mean()
df['category_target_enc'] = df['category'].map(category_mean)

# Embedding (dùng trong deep models)
# → Xem: ../03_ranking/deep_models/DeepFM.md
```

### 2. Temporal features

```python
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Recency: thời gian kể từ lần tương tác cuối
df['days_since_last_interaction'] = (
    df['timestamp'] - df.groupby('user_id')['timestamp'].transform('max')
).dt.days.abs()
```

### 3. Aggregation features (user history)

```python
# Số lần user tương tác với từng category
user_category_count = df.groupby(['user_id', 'category']).size().unstack(fill_value=0)

# CTR của user theo category
user_ctr = df.groupby(['user_id', 'category']).agg(
    clicks=('clicked', 'sum'),
    impressions=('impressed', 'count')
)
user_ctr['ctr'] = user_ctr['clicks'] / user_ctr['impressions']
```

### 4. Sequence features (quan trọng với DIN/DIEN)

```python
# Last N items user đã click (sequence)
user_history = df.sort_values('timestamp').groupby('user_id')['item_id'].apply(
    lambda x: list(x[-50:])  # last 50 items
)
```

## Feature Store

Lưu trữ và phục vụ features hiệu quả:
- **Offline store**: Hive, BigQuery — training
- **Online store**: Redis, DynamoDB — serving real-time
- **Tools**: Feast, Tecton, Vertex AI Feature Store

→ Xem chi tiết: ../../05_serving/feature_store.md

## Anti-patterns cần tránh

```
1. Feature leakage: dùng thông tin từ tương lai để predict quá khứ
2. Train-serve skew: feature tính khác nhau ở train vs serving
3. High cardinality raw: dùng raw user_id không qua embedding
4. Ignore temporal split: shuffle data trước khi split → data leakage
```
