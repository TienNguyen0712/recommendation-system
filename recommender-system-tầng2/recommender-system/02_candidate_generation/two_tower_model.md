# Two-Tower Model

## Kiến trúc tổng quan

```
User features                    Item features
(id, history, context)           (id, category, text)
      |                                 |
  [User Tower]                    [Item Tower]
  MLP layers                      MLP layers
      |                                 |
  user_emb (d-dim)  ←dot product→  item_emb (d-dim)
                          |
                        score
```

Hai tower hoàn toàn tách biệt → có thể precompute item embeddings offline.

## Code (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, user_feat_dim,
                 item_feat_dim, emb_dim=64):
        super().__init__()

        # User tower
        self.user_id_emb = nn.Embedding(num_users, 32)
        self.user_tower = nn.Sequential(
            nn.Linear(32 + user_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

        # Item tower
        self.item_id_emb = nn.Embedding(num_items, 32)
        self.item_tower = nn.Sequential(
            nn.Linear(32 + item_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def user_encode(self, user_id, user_feats):
        uid_emb = self.user_id_emb(user_id)
        x = torch.cat([uid_emb, user_feats], dim=-1)
        return F.normalize(self.user_tower(x), dim=-1)

    def item_encode(self, item_id, item_feats):
        iid_emb = self.item_id_emb(item_id)
        x = torch.cat([iid_emb, item_feats], dim=-1)
        return F.normalize(self.item_tower(x), dim=-1)

    def forward(self, user_id, user_feats, item_id, item_feats):
        u_emb = self.user_encode(user_id, user_feats)
        i_emb = self.item_encode(item_id, item_feats)
        return (u_emb * i_emb).sum(dim=-1)  # dot product score
```

## Training: In-batch Negative Sampling

```python
def train_step(model, batch, temperature=0.05):
    user_id, user_feats, item_id, item_feats = batch
    # batch_size = B

    u_emb = model.user_encode(user_id, user_feats)   # (B, d)
    i_emb = model.item_encode(item_id, item_feats)   # (B, d)

    # Similarity matrix: mỗi user vs TẤT CẢ items trong batch
    logits = torch.matmul(u_emb, i_emb.T) / temperature  # (B, B)

    # Diagonal = positive pairs, off-diagonal = in-batch negatives
    labels = torch.arange(len(user_id))
    loss = F.cross_entropy(logits, labels)
    return loss
```

## Serving: ANN Search với FAISS

```python
import faiss
import numpy as np

# --- Offline: index tất cả item embeddings ---
model.eval()
all_item_embs = []
for item_id, item_feats in item_dataloader:
    emb = model.item_encode(item_id, item_feats)
    all_item_embs.append(emb.detach().numpy())

all_item_embs = np.vstack(all_item_embs).astype('float32')  # (N_items, d)

# Build FAISS index (IVF + flat, tốt cho production)
d = all_item_embs.shape[1]
index = faiss.IndexFlatIP(d)         # Inner Product (= cosine nếu normalized)
index.add(all_item_embs)
faiss.write_index(index, "item_index.faiss")

# --- Online: query top-K cho user ---
index = faiss.read_index("item_index.faiss")

user_emb = model.user_encode(user_id, user_feats)
user_emb = user_emb.detach().numpy().astype('float32')

scores, item_indices = index.search(user_emb, k=100)  # Top-100 candidates
# → chuyển sang Ranking stage
```

## So sánh với CF

| | Matrix Factorization | Two-Tower |
|---|---|---|
| Input features | Chỉ ID | ID + bất kỳ features nào |
| Cold-start | Kém | Tốt hơn (dùng content features) |
| Scalability | Tốt | Rất tốt (precompute item emb) |
| Complexity | Đơn giản | Phức tạp hơn |
| Dùng ở đâu | Baseline | YouTube, Pinterest, Airbnb |

## Paper gốc
- **YouTube DNN (2016):** "Deep Neural Networks for YouTube Recommendations"
- Là paper đầu tiên popularize two-tower cho RecSys ở scale lớn

## Ghi nhớ
> Two-Tower = candidate generation engine hiện đại nhất.
> User tower online (real-time), Item tower offline (precomputed).
> Kết hợp với FAISS/HNSW để tìm Top-K trong milliseconds với hàng triệu items.
