# User-Based Collaborative Filtering

## Ý tưởng

Tìm những user có lịch sử rating giống user hiện tại (neighbors),
rồi gợi ý những item mà neighbors thích nhưng user chưa xem.

```
User A: [5, 4, ?, 3, ?]
User B: [4, 5, ?, 4, ?]  ← giống A → neighbor
User C: [?, ?, 5, ?, 4]  ← khác A

Gợi ý cho A: item mà B đã rate cao mà A chưa xem
```

## Công thức

### Cosine Similarity
```
sim(u, v) = (r_u · r_v) / (||r_u|| × ||r_v||)
```

### Pearson Correlation (có mean-centering)
```
sim(u, v) = Σ(r_ui - r̄_u)(r_vi - r̄_v) / sqrt(Σ(r_ui-r̄_u)² × Σ(r_vi-r̄_v)²)
```
Tốt hơn cosine vì loại bỏ bias (user hay rate cao hoặc thấp)

### Dự đoán rating
```
r̂_ui = r̄_u + Σ sim(u,v)·(r_vi - r̄_v) / Σ|sim(u,v)|
        v ∈ neighbors(u), r_vi known
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

# Tính similarity (chỉ dùng rated items)
R_masked = np.where(R > 0, R, 0)
sim = cosine_similarity(R_masked)  # (n_users, n_users)
np.fill_diagonal(sim, 0)           # không so sánh với chính mình

# Top-K neighbors cho user 0
user_id = 0
k = 2
neighbors = np.argsort(sim[user_id])[::-1][:k]

# Dự đoán rating cho item chưa xem
def predict(user_id, item_id, R, sim, k=2):
    neighbors = np.argsort(sim[user_id])[::-1][:k]
    num, den = 0, 0
    for v in neighbors:
        if R[v, item_id] > 0:
            num += sim[user_id, v] * R[v, item_id]
            den += abs(sim[user_id, v])
    return num / den if den > 0 else 0
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Đơn giản, dễ hiểu | O(U²) memory và compute |
| Không cần item features | Sparse data → similarity kém |
| Gợi ý có thể surprise | Cold-start user mới |
| | Không scale tốt |

## Khi nào dùng
- Dataset nhỏ/vừa (< 100K users)
- Cần baseline nhanh
- Không có item features

## Ghi nhớ
> User-based CF tốt về mặt lý thuyết nhưng không scale.
> Trong production: thay bằng Matrix Factorization hoặc Two-Tower.
