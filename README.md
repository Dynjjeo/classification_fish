# classification_fish
# Báo cáo Phân tích và Phân loại Dữ liệu Hình ảnh Cá

## 1. Rút gọn số chiều dữ liệu và hiển thị trực quan

### Phương pháp:
- Sử dụng kỹ thuật PCA (Principal Component Analysis) và t-SNE (t-distributed Stochastic Neighbor Embedding) để giảm số chiều của đặc trưng ảnh.
- Dữ liệu đầu vào là các ảnh RGB đã được chuyển đổi sang vector đặc trưng (flatten hoặc trích xuất đặc trưng từ mô hình pretrained như VGG16).

### Trực quan:
- Biểu đồ scatter 2D thể hiện sự phân bố các lớp sau khi giảm số chiều bằng PCA/t-SNE.
- Màu sắc biểu diễn các lớp cá khác nhau.

---

## 2. Phân cụm dữ liệu gốc bằng KMeans

### Phương pháp:
- Áp dụng KMeans Clustering (k=9) trên tập dữ liệu ảnh RGB đã trích xuất đặc trưng.
- Sử dụng PCA để giảm chiều trước khi phân cụm.

### Trực quan:
- Scatter plot hiển thị các cụm KMeans với màu sắc riêng biệt cho từng cụm.
- Gán nhãn cụm và so sánh với nhãn thực để đánh giá trực quan.

---

## 3. Phân loại bằng Multinomial Logistic Regression (Softmax)

### Phương pháp:
- Ảnh RGB được chuyển thành vector đầu vào bằng cách flatten hoặc trích đặc trưng từ mô hình VGG16 (Transfer Learning).
- Mô hình huấn luyện: Softmax Regression (Logistic đa lớp).

### Huấn luyện và kiểm tra:
- Training trên tập `Fish_Dataset`.
- Validation/Test trên tập `NA_Fish_Dataset`.

### Kết quả:
- Accuracy trên tập test: ~`XX%`
- Confusion Matrix, Recall, Precision được tính trên từng lớp.

---

## 4. Phân loại bằng Convolutional Neural Network (CNN)

### Kiến trúc mô hình:
```python
Input -> [Conv2D + ReLU + MaxPooling]*3 -> Flatten ->
Dense (256) + ReLU -> Dropout -> Dense (128) + ReLU -> Dense(9) + Softmax
