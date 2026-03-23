# Hệ thống gợi ý (Recommendation System)

## Toàn cảnh 

Xây dựng hệ thống gợi ý tức là quá trình sử dụng các nguồn dữ liệu đầu vào sẵn (Dữ liệu người dùng, sản phẩm, context) kết hợp cùng các phương pháp lọc để có thể đưa ra sản phẩm cho khách hàng, người dùng. Quá trình thường được thực hiện với các lộ trình sau: 

```
Dữ liệu có sẵn -> Phương pháp lọc gơi ý -> Mô hình học máy -> Triển khai 
                                                |
                                              Đánh giá
```

Một pipeline hệ thống gợi ý hoàn chỉnh sẽ bao gồm các bước:

```
Thu thập và xử lý dữ liệu -> Tạo ứng viên -> Xếp hạng -> Lọc lại + Logic nghiệp vụ -> Phục vụ và hạ tầng -> Đánh giá và giám sát 
```

---

## 1. Dữ liệu 

### 1.1. Các loại dữ liệu đầu vào trong hệ thống gợi ý: 

- **Explicit Feedback (Hiếm xảy ra hơn)**: Dữ liệu mà người dùng tự nguyên cung cấp. Rõ ràng, tin cậy nhưng rất khó để thu thập _(Rating, Like/Dislike, Review, Rakinh)_
- **Implicit Feedback**: Dữ liệu mà người dùng không tự nguyên cung cấp. Mà phải thu thập từ các trang web _(Số lần truy cập, Lượt click, Lượt xem)_

Với mỗi bộ loại dữ liệu khác nhau ta lại càng sử dujgn các thuật toán khác nhau: 
- Explitcit: SVD, NMF, SVD++
- Implitcit: ALS, BPR, NCF

### 1.2. Vấn đề cốt lõi của Implicit là: 
- Khi user click hay không click cũng không thể xác định được là do không thích hay không thấy
- Đã mua rồi không càn mua lại
- Thấy nhưng không có thời gian

Do vậy để xử lý những điều trên ta có thể thực hiện các phương pháp

- ALS, BRP, Negative Sampling

### 1.3. Các loại Implicit phổ biến

| Signal | Strenghit | Ghi chú |
| ------- | ------- | ------- | 
| Purchase | Rất mạnh | Ý định rõ ràng nhất | 
| Add to cart | Mạnh | Có quan tâm nhưng chưa mua |
| Click | Trung bình | Nhiều noise (clickbait) | 
| View (>30s) | Trung bình | Lọc bỏ accidental click | 
| Impression | Yếu | Chỉ biết user đã thấy | 
| Search | Mạnh | Intent signal rõ |

### 1.4. Ma trận user-item 

Là một ma trận E có kịch thước (U X I)
- U: Số user, I: Số item
- R[u][i]: rating của user u với item i
- 0 hoặc NaN = chưa tương tác

```
         Inception  Interstellar  Avatar  Tenet  Dune
An           5           4          0       3      0
Bình         4           5          0       4      0
Chi          0           0          5       0      4
Dũng         3           0          4       0      5
```

**Đặc điểm**

#### Sparsity (độ thưa)

```python
sparsity = 1 - (n_interactions / (n_users * n_items))
# Thực tế: sparsity thường > 99%
# Netflix: ~99.8%   Amazon: ~99.9%
```

#### Scalability
```
Netflix:  ~500K users × 17K movies  → 8.5B cells
Spotify:  ~400M users × 80M songs   → 32T cells  (không thể lưu toàn bộ!)
```
→ Phải dùng sparse matrix format (CSR/CSC).

### 1.5. Phân loại features: 

**Theo user**

```
Demographic:    age, gender, location, language
Behavioral:     session length, avg click rate, purchase history
Preference:     genre preference, price range, brand affinity
Temporal:       time of day, day of week, recency of activity
```

```
r_u = vector rating của user u
→ Dùng để tìm user tương tự
→ User-based CF
```

**Theo item**

```
Content:        title, description, category, tags, brand
Metadata:       price, release date, popularity, avg rating
Visual:         thumbnail embedding (CNN), color palette
Textual:        TF-IDF, BERT embedding của description
```

```
r_i = vector rating của item i
→ Dùng để tìm item tương tự
→ Item-based CF
```

**Theo context**

```
Device:         mobile vs desktop, app version
Time:           hour, weekday, season
Location:       city, country
Session:        last N items clicked, current search query
```

**Theo cross features**
```
user × item:    user đã xem category này bao nhiêu lần
user × time:    user thường active vào giờ nào
item × time:    item trending vào thời điểm nào
```

Các kỹ thuật Features Engineering bao gồm
- Encoding categorical features (chuyển thành các đặc trưng sô)
- Temporal features (chuyển ngày giờ tháng năm)
- Aggregation features (user history) -> Chỉnh mean, max, min, ...

 Cần tránh
 - Feature leakage: Dùng towng lai để dự đoán quá khứ
 - Train-serve skew: Tính khác nhau trên train
 - High cardinality raw: dùng raw user_id không qua embedding
 - Ignore temporal split: shuffle data trước khi split → data leakage

---

## 2. Lựa chọn ứng viên 

Phần này là phần cốt lõi cũng như là kiến trúc trọng tâm cho Hệ thống gợi ý, phần này được chia làm ba lớp kỹ thuật tăng dần theo độ phức tạp 

### 2.1. Tầng 1: Collaborative Filtering (CF)

- **User-Based CF:** Tìm những user có lịch sử rating trông giống với user hiện tại -> Dùng rating của họ để dự đoán item chưa xem
  - Cách tính sẽ theo Corr, hoặc cosin similar
  - Nhược điểm: Nặng bộ nhớ khi dữ liệu lớn - Kém chính xác - Khi có user mới hoàn toàn thì khoogn thể tính chính xác
- **Item-Based CF:** Thaty vì nói ai giống user này thì người ta nói phim nào giống phim A. Oỏn định hơn user vì item ít thay đổi hơn
  - Cách tích tương tự như user-base nhưng theo cột
  - Ổn định hơn user  
- **Matrix Factorization:** Mỗi user và item đươc embed vào không gian k chiều ẩn. Rating dự đoán = nhân 2 vecto, học hai ma trận nhỏ thay vi lưu toàn bộ ma trận
  - 3 biến thể quan trọng:
    - SVD++: Dùng khi có explicit rating
    - ALS: Implitcit data
    - BPR: Chỉ biết positive (item đã click phải dược rank cao hơn item chưa click) - Loss là hàm sigmoid
      - Với mỗi positive ta cần smaple 1 negative: Random (Chọn ngẫu nhiên từ tất cả item chưa click) - Popularity (Ưu tiên item phổ biến làm negative) - In-batch (Dùng positive của các user khác trong cùng batch làm negative)

**Thảo luận**

Tại sao cần BRP: 

> Trong thực tế các dữ liệu đa số là implicit - user không rate phim họ chỉ click hoặc không click (Vấn đề là không click không có nghĩa là không thích có thể đơn giản là user chưa nhìn thấy). Ý tưởng cảu BRP chính là dảm bảo item dã tương tác được xếp hạng cao hơn item chưa tương tác


### 2.2. Tầng 2: Content-Based Filtering

#### 2.2.1. TF-IDF

Biểu diễn item (phim, sản phẩm, bài viết) thành vector số dựa trên nội dung text.
Tìm item tương tự = tìm vector gần nhau.
  - Từ xuất hiện nhiều trong toàn corpus (the, is, a) → IDF thấp → ít quan trọng
  - Từ xuất hiện ít nhưng đặc trưng (Nolan, sci-fi) → IDF cao → quan trọng

- **Công thức:**
```
- TF (Term Frequency)
TF(t, d) = số lần từ t xuất hiện trong document d / tổng số từ trong d

- IDF (Inverse Document Frequency)
IDF(t) = log(N / df_t)
N   = tổng số documents
df_t = số documents chứa từ t

- TF-IDF
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Không cần user data | Không capture user preference |
| Giải quyết cold-start cho item mới | Vocabulary mismatch (synonym) |
| Dễ interpret | Không hiểu ngữ nghĩa sâu |
| Nhanh, nhẹ | Kém hơn embedding với large corpus |

> TF-IDF tốt với keyword matching. Embedding tốt với semantic matching.
> Trong production: thường dùng embedding, TF-IDF làm fallback hoặc feature bổ sung.

#### 2.2.2. Embedding Similarity
Biểu diễn user và item thành dense vector trong không gian k chiều. Items/users tương tự nhau → vector gần nhau trong không gian đó.
- Các loại embedding
```python
- Item Embedding
# Từ text description (Sentence-BERT)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = ["sci-fi action Nolan", "romantic comedy Paris", ...]
item_embeddings = model.encode(descriptions)  # shape: (n_items, 384)

- User Embedding
# Cách 1: Average của item embeddings đã tương tác
def user_embedding(user_history, item_embeddings):
    interacted = item_embeddings[user_history]
    return interacted.mean(axis=0)  # shape: (384,)

# Cách 2: Weighted average (gần đây hơn = weight cao hơn)
def weighted_user_embedding(user_history, timestamps, item_embeddings):
    weights = np.exp(-0.1 * (now - timestamps))  # time decay
    weights /= weights.sum()
    return (item_embeddings[user_history] * weights[:, None]).sum(axis=0)

- ID Embedding (Learned)
import torch.nn as nn

# Học embedding từ interaction data (dùng trong Two-Tower, MF)
user_emb = nn.Embedding(num_users, dim=64)
item_emb = nn.Embedding(num_items, dim=64)

# Prediction: dot product
score = (user_emb(user_id) * item_emb(item_id)).sum()
```

- Đo độ tương đồng

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

- Content-Based pipeline đầy đủ

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

| Loại | Ưu điểm | Nhược điểm |
|------|---------|-----------|
| TF-IDF vector | Nhanh, explainable | Sparse, no semantics |
| Pretrained (BERT) | Semantic rich | Tốn compute |
| Learned ID emb | Capture interaction patterns | Cold-start |
| Hybrid | Tốt nhất | Phức tạp hơn |

> Item embedding từ content → giải quyết item cold-start.
> User embedding = trung bình item embeddings đã tương tác → giải quyết user cold-start một phần.
> Kết hợp content embedding + ID embedding = tốt nhất trong production.

- **Consin Similarity:**

### Tầng 3: Two-Tower + ANN Search 

- **Two Tower Model:**
- **ANN Search:**





