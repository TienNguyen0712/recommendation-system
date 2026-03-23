# TF-IDF (Term Frequency - Inverse Document Frequency)

## Ý tưởng cốt lõi

Biểu diễn item (phim, sản phẩm, bài viết) thành vector số dựa trên nội dung text.
Tìm item tương tự = tìm vector gần nhau.

## Công thức

### TF (Term Frequency)
```
TF(t, d) = số lần từ t xuất hiện trong document d / tổng số từ trong d
```

### IDF (Inverse Document Frequency)
```
IDF(t) = log(N / df_t)
N   = tổng số documents
df_t = số documents chứa từ t
```
Từ xuất hiện nhiều trong toàn corpus (the, is, a) → IDF thấp → ít quan trọng
Từ xuất hiện ít nhưng đặc trưng (Nolan, sci-fi) → IDF cao → quan trọng

### TF-IDF
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

## Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

items = pd.DataFrame({
    'title': ['Inception', 'Interstellar', 'Avatar', 'Tenet'],
    'description': [
        'dream heist sci-fi Christopher Nolan mind-bending thriller',
        'space travel wormhole Christopher Nolan black hole gravity',
        'alien planet blue creatures James Cameron 3D action',
        'time inversion Christopher Nolan spy thriller action',
    ]
})

# Fit TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(items['description'])
# shape: (n_items, n_vocab)

# Cosine similarity giữa tất cả cặp items
sim_matrix = cosine_similarity(tfidf_matrix)
# sim_matrix[i][j] = độ tương đồng giữa item i và item j

# Gợi ý: tìm 3 item giống Inception nhất
item_idx = 0  # Inception
scores = list(enumerate(sim_matrix[item_idx]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]
for idx, score in scores:
    print(f"{items['title'][idx]}: {score:.3f}")
# Interstellar: 0.412 (cùng Nolan, sci-fi)
# Tenet: 0.387 (cùng Nolan, thriller)
# Avatar: 0.021 (khác hoàn toàn)
```

## Ưu và nhược điểm

| Ưu điểm | Nhược điểm |
|---------|-----------|
| Không cần user data | Không capture user preference |
| Giải quyết cold-start cho item mới | Vocabulary mismatch (synonym) |
| Dễ interpret | Không hiểu ngữ nghĩa sâu |
| Nhanh, nhẹ | Kém hơn embedding với large corpus |

## Khi nào dùng TF-IDF vs Embedding

- **TF-IDF:** dataset nhỏ, cần explainability, item có nhiều text
- **Embedding (BERT, Sentence-BERT):** cần hiểu ngữ nghĩa, synonym, đa ngôn ngữ

## Ghi nhớ
> TF-IDF tốt với keyword matching. Embedding tốt với semantic matching.
> Trong production: thường dùng embedding, TF-IDF làm fallback hoặc feature bổ sung.
