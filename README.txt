# Mô tả dữ liệu (thư mục data): 

Bộ dữ liệu chứa 9 loại hải sản khác nhau được thu nhập từ một siêu thị Izmir, Thổ Nhị Kì được xuất bản trong ASYU 2020.

Trích dẫn:
@inproceedings{ulucan2020large,
  title={A Large-Scale Dataset for Fish Segmentation and Classification},
  author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={1--5},
  year={2020},
  organization={IEEE}
}

Bộ dữ liệu bao gồm:
- gilt head bream (cá tráp đầu vàng)
- red sea bream (cá tráp đỏ)
- sea bass (cá vược)
- red mullet (cá đối đỏ)
- horse mackerel (cá thu ngựa)
- black sea sprat (cá trích biển đen)
- striped red mullet (cá đối đỏ sọc)
- trout (cá hồi)
- shrimp image (tôm)

Tập dữ liệu có kích thước 590x445, tất cả các nhãn dữ liệu đã được tăng cường (bằng cách lật và xoay ảnh) tổng số hình ảnh cho mỗi lớp là 2000; 1000 hình ảnh cá RGB và 1000 ground truth theo cặp của chúng.

## Fish_Dataset:
- Mỗi lớp được lưu trữ trong một tệp riêng, trong mỗi tệp bao gồm 2 thư mục con chứa: 1000 ảnh RGB và 1000 nhãn Ground Truth.
- Tất cả các hình ảnh cho mỗi lớp được sắp xếp từ "00000.png" đến "01000.png".
## NA_Fish_Dataset:
- Tập dữ liệu được sử dụng để kiểm tra mô hình, với mỗi loại cá có 50 ảnh, tôm có 30 ảnh.
- Tất cả ảnh được đặt tên từ "00001.png" đến "00050.png"

# Mô tả cấu trúc thư mục:
## Data:
+---data: mỗi thư mục con chứa 9 thư mục đại diện cho 9 lớp cá: Black Sea Sprat, Gilt Head Bream, Hourse Mackerel, Red Mullet, Red Sea Bream, Sea Bass,
|	  Shrimp, Striped Red Mullet, Trout.
|   +---data_fix
|   |   +---Fish_Dataset_GT: ảnh Ground Truth của 9000 ảnh train.
|   |   |
|   |   +---Fish_Dataset_Segment: ảnh sau khi đã thực hiện phân đoạn cho 9000 ảnh train.
|   |   |
|   |   +---NA_Fish_Dataset_GT: ảnh Ground Truth của 430 ảnh test.
|   |   |
|   |   \---NA_Fish_Dataset_Segment: ảnh sau khi đã thực hiện phân đoạn cho 430 ảnh test.
|   |
|   \---data_raw
|       +---Fish_Dataset: bộ dữ liệu gốc bao gồm 9000 ảnh train.
|       |
|       +---NA_Fish_Dataset: bộ dữ liệu test sau khi đã chuẩn hóa định dạng và tên.
|       |
|       \---NA_Fish_Dataset_Raw: bộ dữ liệu gồm 430 ảnh test: 30 ảnh Trout và 50 ảnh mỗi loại cá còn lại.

## Model:
+---model
    +---classification_cnn: mạng Convolution Neural Network phân loại cá.
    |       cnn.ipynb: code.
    |       model1.hdf5: file lưu trọng số tốt nhất của CNN phân loại cá.
    |       model2.hdf5
    |       model3.hdf5
    |
    +---classification_logistic: mạng MLR phân loại cá.
    |       soft_max.ipynb: MLR sử dụng Soft Max trên dữ liệu gốc.
    |       soft_max_transfer.ipynb: MLR kết hợp với kỹ thuật transfer learning dùng để trích xuất đặc trưng.
    |
    \---clustering_kmeans
           kmeans.ipynb: thuật toán KMeans trên dữ liệu gốc.
           kmeans_segment.ipynb: thuật toán KMeans trên dữ liệu đã được phân đoạn.
           kmeans_segment_transfer.ipynb: thuật toán KMeans trên dữ liệu gốc kết hợp với kĩ thuật transfer learning trích xuất đặc trưng.
           kmeans_transfer_learning.ipynb: thuật toán KMeans trên dữ liệu đã phân đoạn kết hợp với kỹ thuật transsfer learning trích xuất đặc trưng.

## Preprocessing:
+---preprocessing
    +---preprocess
    |       describe.ipynb: mô tả và trực quan dữ liệu 2D và 3D.
    |       fix_image.ipynb: sửa tên và định dạng ảnh.
    |       segment.ipynb: phân đoạn hình ảnh.
    |
    \---segmeantation
            unet.ipynb: code xây dựng mô hình unet dùng để phân đoạn.
            unet_1.hdf5: file lưu trọng số mô hình unet.