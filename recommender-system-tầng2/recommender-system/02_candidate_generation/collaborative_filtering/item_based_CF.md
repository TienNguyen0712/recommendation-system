# Item-Based Collaborative Filtering

## Ý tưởng

Thay vì hỏi "ai giống user này?", hỏi "item nào giống item A?"
Tính similarity theo cột (item vectors) thay vì hàng (user vectors).

```
         Inception  Interstellar  Avatar  Tenet
User A:     5           4           0       3
User B:     4           5           0       4
User C:     0           0           5       0

sim(Inception, Interstellar) = cosine([5,4,0], [4,5,0]) ≈ 0.98  ← rất giống
sim(Inception, Avatar)       = cosine([5,4,0], [0,0,5]) = 0.00  ← khác hoàn toàn
```

## Công thức

```
sim(i, j) = cosine(col_i, col_j)

r̂_ui = Σ sim(i,j) × r_uj / Σ |sim(i,j)|
        j rated by u
```

## Code

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

R = np.array([
    [5, 4, 0, 3, 0],
    [4, 5, 0, 4, 0],
    [0, 0, 5, 0, 4],
    [3, 0, 4, 0, 5],
], dtype=float)

# Tính item-item similarity (transpose để tính theo cột)
R_T = R.T  # (n_items, n_users)
item_sim = cosine_similarity(R_T)  # (n_items, n_items)
np.fill_diagonal(item_sim, 0)

# Gợi ý cho user 0: tìm item giống những gì họ đã xem
def recommend_item_based(user_id, R, item_sim, n_rec=3):
    user_ratings = R[user_id]
    rated = np.where(user_ratings > 0)[0]
    unrated = np.where(user_ratings == 0)[0]

    scores = {}
    for item in unrated:
        num, den = 0, 0
        for rated_item in rated:
            num += item_sim[item, rated_item] * user_ratings[rated_item]
            den += abs(item_sim[item, rated_item])
        scores[item] = num / den if den > 0 else 0

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]
```

## Ưu điểm so với User-Based

| | User-Based | Item-Based |
|---|---|---|
| Stability | Users thay đổi thường xuyên | Items ổn định hơn |
| Precompute | Khó (user profile thay đổi) | Dễ (item sim tính offline) |
| Scale | Kém (O(U²)) | Tốt hơn (O(I²), I thường < U) |
| Amazon 2003 | Không | Có — đây là paper nổi tiếng |

## Precompute strategy (production)

```python
# Tính item-item similarity offline (batch job hàng ngày)
# Lưu vào database: (item_i, item_j, similarity)

# Online serving: chỉ lookup, không compute
def get_candidates(user_id, top_k=100):
    user_history = get_user_history(user_id)  # items đã xem
    candidates = set()
    for item_id in user_history[-10:]:  # chỉ dùng 10 item gần nhất
        similar = db.query(f"SELECT item_j FROM item_sim WHERE item_i={item_id} ORDER BY sim DESC LIMIT 20")
        candidates.update(similar)
    return list(candidates)[:top_k]
```

## Ghi nhớ
> Item-based CF ổn định hơn, precompute được → phổ biến hơn trong production.
> Amazon vẫn dùng biến thể của item-based CF đến ngày nay.
> Nhược điểm chính: vẫn bị cold-start với item mới.
