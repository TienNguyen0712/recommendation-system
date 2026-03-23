# Explicit vs Implicit Feedback

## Định nghĩa

| | Explicit | Implicit |
|---|---|---|
| Ví dụ | Rating 1-5 sao, Like/Dislike | Click, View, Purchase, Time-on-page |
| Ý nghĩa | Rõ ràng, user chủ động | Nhiễu, phải suy luận |
| Độ phổ biến | Hiếm | Rất phổ biến |
| Ứng dụng | Netflix, Goodreads | TikTok, Shopee, YouTube |

## Vấn đề với Implicit data

- Không click ≠ không thích (có thể chưa thấy)
- Click ≠ thích (click nhầm, clickbait)
- Time-on-page cao có thể do khó đọc, không phải hay
- **Assumption an toàn nhất:** đã tương tác > chưa tương tác (partial order)

## Cách xử lý

### Với Explicit:
- Dùng trực tiếp làm target trong MSE loss
- Cần xử lý missing values

### Với Implicit:
- **ALS:** gán confidence c_ui = 1 + α·freq_ui, không bỏ qua zero
- **BPR:** sample cặp (positive, negative), tối ưu ranking thay vì regression

## Ghi nhớ nhanh
> "Explicit = user nói thật. Implicit = user hành động, ta phải đoán ý."
