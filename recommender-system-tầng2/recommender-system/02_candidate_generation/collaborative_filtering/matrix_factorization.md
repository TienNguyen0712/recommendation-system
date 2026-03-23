# Matrix Factorization

## Ý tưởng cốt lõi

Phân tích ma trận rating thưa R(U×I) thành 2 ma trận nhỏ:

```
R  ≈  P  ×  Q^T
(U×I)  (U×k)  (k×I)

P: user embedding matrix  — mỗi hàng = 1 user vector
Q: item embedding matrix  — mỗi hàng = 1 item vector
k: số chiều latent (hyperparameter, thường 32–256)
```

Ý nghĩa các chiều latent (k): có thể capture "thể loại", "phong cách", "độ phức tạp"
— không nhất thiết có tên gọi rõ ràng, model tự học.

## Biến thể 1: SVD — Explicit Ratings

```python
from sklearn.decomposition import TruncatedSVD
import numpy as np

# R_filled: ma trận đã fill missing values (bằng mean)
svd = TruncatedSVD(n_components=50)
P = svd.fit_transform(R_filled)   # user embeddings (U, 50)
Q = svd.components_.T             # item embeddings (I, 50)

# Predict rating
def predict(user_id, item_id):
    return P[user_id] @ Q[item_id]
```

## Biến thể 2: ALS — Implicit Feedback

```
Objective:
L = Σ c_ui (r_ui - p_u · q_i)²  +  λ(||P||² + ||Q||²)

c_ui = 1 + α · freq_ui   (confidence: tương tác nhiều = tin tưởng hơn)
r_ui = 1 nếu đã tương tác, 0 nếu không
```

```python
import implicit
import scipy.sparse as sp

# user_items: sparse matrix (users × items), value = frequency
user_items = sp.csr_matrix(interaction_matrix)

model = implicit.als.AlternatingLeastSquares(
    factors=64,      # k: số chiều latent
    regularization=0.01,
    iterations=20,
    alpha=40         # α trong c_ui = 1 + α*freq
)
model.fit(user_items)

# Gợi ý top-10 cho user 0
ids, scores = model.recommend(0, user_items[0], N=10)
```

## Biến thể 3: BPR — Bayesian Personalized Ranking ⭐

Dùng khi: chỉ biết positive (item đã click), cần tối ưu ranking.

```
Loss: L = -Σ ln σ(r̂_ui - r̂_uj)  +  λ||Θ||²
          (u,i,j) ∈ D_S

i = positive item (đã click)
j = negative item (sampled, chưa click)
σ = sigmoid function
```

```python
import implicit

model = implicit.bpr.BayesianPersonalizedRanking(
    factors=64,
    learning_rate=0.01,
    regularization=0.01,
    iterations=100
)
model.fit(user_items)

# Hoặc dùng LightFM (hỗ trợ cả content features)
from lightfm import LightFM

model = LightFM(loss='bpr', no_components=64)
model.fit(interactions, epochs=30, num_threads=4)
```

## Negative Sampling (quan trọng với BPR)

```python
def sample_negative(user_id, user_positives, n_items):
    while True:
        j = np.random.randint(n_items)
        if j not in user_positives:
            return j

# In-batch negatives (hiệu quả hơn)
# Positive của user khác trong batch = negative cho user hiện tại
```

## So sánh 3 biến thể

| | SVD | ALS | BPR |
|---|---|---|---|
| Data type | Explicit | Implicit | Implicit |
| Objective | MSE | Weighted MSE | Ranking |
| Tối ưu | Closed-form | Alternating | SGD |
| Scale | Trung bình | Tốt (Spark) | Tốt |
| Thư viện | scikit-learn | implicit | implicit, LightFM |

## Ghi nhớ
> SVD cho explicit rating (ít gặp).
> ALS cho implicit data có frequency information (nghe nhạc, xem phim).
> BPR cho implicit data sparse, cần optimize ranking trực tiếp (e-commerce click).
> Two-Tower model = deep learning version của Matrix Factorization.
