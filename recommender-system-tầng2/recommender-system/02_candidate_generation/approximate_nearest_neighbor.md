# Approximate Nearest Neighbor (ANN) Search

## Vấn đề

Với 10M items, mỗi item là vector 64 chiều:
- Brute-force: 10M × 64 phép nhân → ~50ms/query → quá chậm cho production
- ANN: chấp nhận recall@100 ≈ 95% để đổi lấy tốc độ < 5ms

## Các thuật toán chính

### 1. FAISS (Facebook AI Similarity Search)

```python
import faiss
import numpy as np

d = 64          # embedding dimension
n = 1_000_000   # số items

embeddings = np.random.randn(n, d).astype('float32')

# --- IVF (Inverted File Index) ---
# Chia không gian thành n_clusters cụm, chỉ search trong cụm gần nhất
n_clusters = 1000
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)   # học cluster centroids
index.add(embeddings)
index.nprobe = 50         # search trong 50 cụm gần nhất (tradeoff recall vs speed)

# Query
query = np.random.randn(1, d).astype('float32')
scores, indices = index.search(query, k=100)

# --- IVF + PQ (Product Quantization) — tiết kiệm memory ---
index = faiss.IndexIVFPQ(quantizer, d, n_clusters, 8, 8)
# 8 sub-vectors, mỗi sub-vector quantized thành 8 bits → compress 32x
```

### 2. HNSW (Hierarchical Navigable Small World)

```python
# Dùng thư viện hnswlib
import hnswlib

dim = 64
num_elements = 1_000_000

# Build index
p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=200, M=16)
p.add_items(embeddings)
p.set_ef(50)   # ef khi query, càng cao càng chính xác (và chậm hơn)

# Query
labels, distances = p.knn_query(query, k=100)

# Save/load
p.save_index("hnsw_index.bin")
p.load_index("hnsw_index.bin")
```

### 3. ScaNN (Google)

```python
import scann

searcher = scann.scann_ops_pybind.builder(embeddings, 10, "dot_product") \
    .tree(num_leaves=1000, num_leaves_to_search=100) \
    .score_ah(2, anisotropic_quantization_threshold=0.2) \
    .reorder(100) \
    .build()

neighbors, distances = searcher.search_batched(queries)
```

## So sánh

| | FAISS IVF | HNSW | ScaNN |
|---|---|---|---|
| Recall@10 | ~95% | ~99% | ~98% |
| QPS (1M vectors) | ~5000 | ~3000 | ~7000 |
| Memory | Thấp (với PQ) | Cao | Trung bình |
| GPU support | Có | Không | Không |
| Dễ dùng | Trung bình | Dễ | Trung bình |
| Dùng ở đâu | Meta, nhiều nơi | Weaviate, Vespa | Google |

## Vector Databases (production)

Khi cần managed solution với filter + update real-time:

| DB | Đặc điểm |
|----|---------|
| Pinecone | Fully managed, dễ dùng nhất |
| Weaviate | Open-source, hỗ trợ hybrid search |
| Qdrant | Rust-based, nhanh, open-source |
| Milvus | Scale lớn, Alibaba |

## Tradeoff cần nhớ

```
Recall cao  ←────────────────────→  Latency thấp
(nprobe lớn, ef lớn)              (nprobe nhỏ, ef nhỏ)

Thường target: recall@100 ≥ 95%, latency ≤ 10ms
```

## Ghi nhớ
> ANN là cầu nối giữa Two-Tower model và production serving.
> HNSW = chất lượng tốt nhất. FAISS = linh hoạt nhất, GPU support.
> Vector DB = khi cần filter theo metadata + real-time update.
