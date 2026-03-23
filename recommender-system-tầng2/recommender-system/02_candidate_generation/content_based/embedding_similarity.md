# Embedding Similarity

## Ý tưởng

Biểu diễn user và item thành dense vector trong không gian k chiều.
Items/users tương tự nhau → vector gần nhau trong không gian đó.

## Các loại embedding

### Item Embedding
```python
# Từ text description (Sentence-BERT)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = ["sci-fi action Nolan", "romantic comedy Paris", ...]
item_embeddings = model.encode(descriptions)  # shape: (n_items, 384)
```

### User Embedding
```python
# Cách 1: Average của item embeddings đã tương tác
def user_embedding(user_history, item_embeddings):
    interacted = item_embeddings[user_history]
    return interacted.mean(axis=0)  # shape: (384,)

# Cách 2: Weighted average (gần đây hơn = weight cao hơn)
def weighted_user_embedding(user_history, timestamps, item_embeddings):
    weights = np.exp(-0.1 * (now - timestamps))  # time decay
    weights /= weights.sum()
    return (item_embeddings[user_history] * weights[:, None]).sum(axis=0)
```

### ID Embedding (Learned)
```python
import torch.nn as nn

# Học embedding từ interaction data (dùng trong Two-Tower, MF)
user_emb = nn.Embedding(num_users, dim=64)
item_emb = nn.Embedding(num_items, dim=64)

# Prediction: dot product
score = (user_emb(user_id) * item_emb(item_id)).sum()
```

## Đo độ tương đồng

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cosine similarity (phổ biến nhất)
sim = cosine_similarity(vec_a.reshape(1,-1), vec_b.reshape(1,-1))[0][0]

# Dot product (dùng trong Two-Tower sau khi normalize)
sim = np.dot(vec_a, vec_b)

# Euclidean distance (ít phổ biến hơn)
dist = np.linalg.norm(vec_a - vec_b)
```

## Content-Based pipeline đầy đủ

```
Item metadata (text, category, tags)
    → Text Encoder (TF-IDF / BERT)
    → Item Embedding Matrix  (I × d)

User history (clicked items)
    → Average Item Embeddings
    → User Embedding  (d,)

User Embedding × Item Embedding^T
    → Similarity scores
    → Top-K items
```

## So sánh các loại embedding

| Loại | Ưu điểm | Nhược điểm |
|------|---------|-----------|
| TF-IDF vector | Nhanh, explainable | Sparse, no semantics |
| Pretrained (BERT) | Semantic rich | Tốn compute |
| Learned ID emb | Capture interaction patterns | Cold-start |
| Hybrid | Tốt nhất | Phức tạp hơn |

## Ghi nhớ
> Item embedding từ content → giải quyết item cold-start.
> User embedding = trung bình item embeddings đã tương tác → giải quyết user cold-start một phần.
> Kết hợp content embedding + ID embedding = tốt nhất trong production.
