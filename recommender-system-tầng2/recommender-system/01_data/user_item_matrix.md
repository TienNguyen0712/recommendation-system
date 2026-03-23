# User-Item Matrix

## Khái niệm

Ma trận R có kích thước (U × I):
- U = số users, I = số items
- R[u][i] = rating / interaction của user u với item i
- 0 hoặc NaN = chưa tương tác

```
         Inception  Interstellar  Avatar  Tenet  Dune
An           5           4          0       3      0
Bình         4           5          0       4      0
Chi          0           0          5       0      4
Dũng         3           0          4       0      5
```

## Đặc điểm quan trọng

### Sparsity (độ thưa)
```python
sparsity = 1 - (n_interactions / (n_users * n_items))
# Thực tế: sparsity thường > 99%
# Netflix: ~99.8%   Amazon: ~99.9%
```

### Scalability
```
Netflix:  ~500K users × 17K movies  → 8.5B cells
Spotify:  ~400M users × 80M songs   → 32T cells  (không thể lưu toàn bộ!)
```
→ Phải dùng sparse matrix format (CSR/CSC).

## Biểu diễn trong code

```python
import scipy.sparse as sp
import numpy as np

# Cách 1: COO format (tốt cho build)
rows = [0, 0, 1, 1, 2, 2]  # user indices
cols = [0, 1, 0, 1, 2, 4]  # item indices
data = [5, 4, 4, 5, 5, 4]  # ratings

R = sp.coo_matrix((data, (rows, cols)), shape=(4, 5))

# Cách 2: Chuyển sang CSR (tốt cho tính toán)
R_csr = R.tocsr()

# Truy cập user 0
user_0_ratings = R_csr[0]

# Sparsity
sparsity = 1 - R.nnz / (R.shape[0] * R.shape[1])
print(f"Sparsity: {sparsity:.2%}")
```

## Hai góc nhìn

### User-based view (hàng)
```
r_u = vector rating của user u
→ Dùng để tìm user tương tự
→ User-based CF
```

### Item-based view (cột)
```
r_i = vector rating của item i
→ Dùng để tìm item tương tự
→ Item-based CF
```

## Sparse → Dense qua Matrix Factorization
```
R (sparse, U×I)  ≈  P (U×k) × Q^T (k×I)
k << min(U, I)  → compact representation
→ Xem chi tiết: ../collaborative_filtering/matrix_factorization.md
```
