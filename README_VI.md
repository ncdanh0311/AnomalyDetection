<p align="center">
  <img src="./assets/social_anomaly_clustering_animated_logo.gif" alt="Social Anomaly Detection Logo" width="150">
</p>

<h1 align="center">Phát hiện tài khoản bất thường trên mạng xã hội</h1>
<h3 align="center">Isolation Forest kết hợp đặc trưng đồ thị</h3>

<p align="center">
  <b>Đồ án Data Mining | K-Means | DBSCAN | Graph Feature Engineering | Isolation Forest</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/NetworkX-2C7FB8?style=for-the-badge" alt="NetworkX" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

<div align="center">
  <b><a href="./README.md">English Version</a></b>
</div>

---

## Mục lục

- [1. Tổng quan dự án](#1-tổng-quan-dự-án)
- [2. Mục tiêu dự án](#2-mục-tiêu-dự-án)
- [3. Dataset](#3-dataset)
- [4. Phương pháp thực hiện](#4-phương-pháp-thực-hiện)
- [5. Thuật toán baseline cũ](#5-thuật-toán-baseline-cũ)
- [6. Graph Feature Engineering](#6-graph-feature-engineering)
- [7. Phát hiện bất thường bằng Isolation Forest](#7-phát-hiện-bất-thường-bằng-isolation-forest)
- [8. Rule-Based Suspicious Score](#8-rule-based-suspicious-score)
- [9. So sánh thuật toán](#9-so-sánh-thuật-toán)
- [10. Kết quả trực quan hóa](#10-kết-quả-trực-quan-hóa)
- [11. Cấu trúc project](#11-cấu-trúc-project)
- [12. Quy trình chạy notebook](#12-quy-trình-chạy-notebook)
- [13. Cách chạy](#13-cách-chạy)
- [14. Output mong đợi](#14-output-mong-đợi)
- [15. Ghi chú và hạn chế](#15-ghi-chú-và-hạn-chế)
- [16. Kết luận](#16-kết-luận)

---

## 1. Tổng quan dự án

Project này phát hiện các tài khoản bất thường trên mạng xã hội bằng một pipeline Data Mining. Đây là phiên bản nâng cấp từ project clustering cũ, vốn sử dụng **K-Means** và **DBSCAN** để tìm hành vi người dùng bất thường.

Tên đề tài mới:

```text
Phát hiện tài khoản bất thường trên mạng xã hội bằng Isolation Forest kết hợp đặc trưng đồ thị
```

Phiên bản nâng cấp vẫn giữ cấu trúc project cũ, các notebook cũ, code preprocessing, các bước feature engineering, scaling, PCA visualization và các thuật toán clustering baseline. Phần đóng góp mới là pipeline phát hiện bất thường bổ sung dựa trên:

- Graph feature engineering
- Isolation Forest
- Rule-based suspicious scoring đơn giản
- So sánh với K-Means và DBSCAN

Đây là một **đồ án Data Mining**, không phải project Deep Learning hay NLP phức tạp. Project không sử dụng Deep Learning, XGBoost, BERT, LSTM hoặc transformer models.

---

## 2. Mục tiêu dự án

Các mục tiêu chính:

- Giữ lại project clustering gốc và dùng làm baseline.
- Tái sử dụng logic load dữ liệu, preprocessing, feature engineering, scaling, PCA, evaluation và visualization đã có.
- Xây dựng user interaction graph từ metadata mạng xã hội hiện có.
- Trích xuất graph-based features cho từng user.
- Huấn luyện Isolation Forest cho bài toán phát hiện bất thường không giám sát.
- Thêm suspicious score đơn giản dựa trên luật để anomaly score dễ giải thích hơn.
- So sánh K-Means, DBSCAN và Isolation Forest trong cùng một pipeline nhất quán.
- Tạo CSV outputs, trained models và visualizations phục vụ báo cáo đồ án Data Mining.

---

## 3. Dataset

Project sử dụng dataset dạng Twitter/X bot detection.

File raw chính:

```text
datasets/raw/bot_detection_data.csv
```

Các file processed từ pipeline cũ:

```text
datasets/cleaned/bot_detection_clean.csv
datasets/processed/user_features.csv
datasets/processed/user_features_scaled.csv
datasets/processed/user_features_with_label.csv
```

### 3.1 Các cột chính

| Cột | Mô tả | Cách sử dụng |
| --- | --- | --- |
| `user_id` | Mã định danh duy nhất của user | Node ID và mapping kết quả |
| `username` | Tên tài khoản | Dùng để tạo username length |
| `tweet` | Nội dung tweet | Dùng để tạo tweet length |
| `retweet_count` | Số lượt retweet | Behavior feature |
| `mention_count` | Số lượt mention | Behavior feature |
| `follower_count` | Số follower | Behavior feature |
| `verified` | Trạng thái xác minh tài khoản | Behavior feature |
| `hashtags` | Hashtag/token text | Hashtag similarity và hashtag features |
| `created_at` | Thông tin ngày/giờ | Account age feature |
| `bot_label` | Nhãn có sẵn | Chỉ dùng để đánh giá, không dùng khi train unsupervised |

Cột label chỉ được dùng sau khi mô hình đã train để đánh giá kết quả. Các mô hình không giám sát không sử dụng `bot_label` trong quá trình fit.

---

## 4. Phương pháp thực hiện

Project nâng cấp đi theo pipeline Data Mining sau:

```text
Raw dataset
    -> Data cleaning
    -> Feature engineering
    -> Feature scaling
    -> Baseline clustering với K-Means và DBSCAN
    -> Graph construction
    -> Graph feature extraction
    -> Isolation Forest anomaly detection
    -> Rule-based suspicious score
    -> Algorithm comparison
    -> Visualization và reporting
```

Workflow clustering cũ được giữ lại để có thể so sánh phương pháp mới với project ban đầu.

---

## 5. Thuật toán baseline cũ

Các notebook gốc được giữ lại:

```text
notebooks/01_explore_dataset.ipynb
notebooks/02_preprocessing_feature_engineering.ipynb
notebooks/03_kmeans_clustering.ipynb
notebooks/04_dbscan_clustering.ipynb
notebooks/05_evaluation_visualization_updated.ipynb
```

### K-Means

K-Means gom nhóm user thành các cụm hành vi. Những user nằm xa tâm cụm được xem là anomaly tiềm năng.

Output chính:

```text
results/csv/kmeans_clustered_users.csv
results/csv/kmeans_anomaly_users.csv
results/models/kmeans_model.pkl
```

### DBSCAN

DBSCAN xác định các vùng dữ liệu có mật độ cao và đánh dấu các điểm ở vùng mật độ thấp là noise. Các điểm noise được xem là anomaly tiềm năng.

Output chính:

```text
results/csv/dbscan_clustered_users.csv
results/csv/dbscan_anomaly_users.csv
results/models/dbscan_model.pkl
```

Hai thuật toán này vẫn quan trọng vì đóng vai trò baseline để đánh giá liệu pipeline Isolation Forest kết hợp graph features có tạo ra cải tiến rõ ràng hay không.

---

## 6. Graph Feature Engineering

Notebook mới:

```text
notebooks/06_graph_feature_engineering.ipynb
```

Mỗi user được biểu diễn như một node trong graph.

Dataset không chứa quan hệ user-user trực tiếp như `user A mentions user B` hoặc `user A retweets user B`. Vì vậy, graph được xây dựng từ các quan hệ thay thế có thể suy ra từ dataset:

- `feature_similarity`: cosine similarity giữa các behavior features đã scale.
- `hashtag_similarity`: user có hashtag/token text tương tự nhau.
- `interaction_similarity`: user có `mention_count`, `retweet_count` và `hashtag_count` tương tự nhau.

### 6.1 Graph features được tạo

Notebook tính các graph features sau cho từng user:

| Feature | Ý nghĩa |
| --- | --- |
| `degree` | Số kết nối graph của user |
| `degree_centrality` | Độ trung tâm tương đối dựa trên degree |
| `clustering_coefficient` | Mức độ user thuộc về các local graph neighborhoods |
| `interaction_score` | Weighted degree / interaction strength đã chuẩn hóa |
| `pagerank` | Điểm quan trọng của node dựa trên cấu trúc liên kết |

Output:

```text
datasets/processed/user_graph_features.csv
```

---

## 7. Phát hiện bất thường bằng Isolation Forest

Notebook mới:

```text
notebooks/07_isolation_forest_anomaly_detection.ipynb
```

Isolation Forest được sử dụng làm thuật toán phát hiện bất thường chính trong phần nâng cấp. Thuật toán này phù hợp với project vì:

- Là phương pháp không giám sát.
- Hoạt động tốt với các numerical behavior features.
- Phát hiện các quan sát dễ bị cô lập khỏi phần lớn dữ liệu.
- Đơn giản và phù hợp với đồ án Data Mining hơn các mô hình Deep Learning.

### 7.1 Bộ feature sử dụng

Isolation Forest sử dụng cả behavior features cũ và graph features mới.

Behavior features cũ:

```text
retweet_count
mention_count
follower_count
verified
tweet_length
username_length
hashtag_count
has_hashtag
account_age_days
```

Graph features mới:

```text
degree
degree_centrality
clustering_coefficient
interaction_score
pagerank
```

### 7.2 Các cột output

File kết quả Isolation Forest gồm:

| Cột | Ý nghĩa |
| --- | --- |
| `isolation_score` | Isolation Forest anomaly score đã chuẩn hóa |
| `anomaly_score` | Alias cho Isolation Forest score |
| `anomaly_label` | Nhãn Isolation Forest, `-1 = anomaly`, `1 = normal` |
| `suspicious_score` | Điểm nghi ngờ dựa trên luật |
| `final_anomaly_score` | Điểm anomaly tổng hợp |
| `final_anomaly_label` | Nhãn cuối cùng, `-1 = anomaly`, `1 = normal` |
| `is_anomaly` | Nhãn binary dùng để so sánh, `1 = anomaly`, `0 = normal` |

Output chính:

```text
results/csv/isolation_forest_results.csv
results/models/isolation_forest.pkl
```

---

## 8. Rule-Based Suspicious Score

Project bổ sung một suspicious score đơn giản để final anomaly score dễ giải thích hơn.

Điểm này không phải là mô hình machine learning phức tạp. Nó dựa trên các luật hành vi trực tiếp như:

- Hashtag count rất cao.
- Hashtag ratio rất cao.
- Mention count rất cao.
- Retweet count rất cao.
- Retweet activity cao so với follower count.
- Follower thấp nhưng activity cao.
- Graph interaction score rất cao.

Mỗi điều kiện đúng sẽ cộng một điểm nghi ngờ. Sau đó điểm được chuẩn hóa về `[0, 1]`.

Điểm cuối cùng:

```text
final_anomaly_score = 0.7 * isolation_score + 0.3 * suspicious_score
```

Cách này giữ mô hình đơn giản nhưng vẫn bổ sung được các tín hiệu hành vi có ý nghĩa miền bài toán.

---

## 9. So sánh thuật toán

Notebook mới:

```text
notebooks/08_compare_algorithms.ipynb
```

Notebook này so sánh:

- K-Means
- DBSCAN
- Isolation Forest kết hợp graph features

### 9.1 Metrics khi có label

Nếu có `bot_label`, notebook tính:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 9.2 Metrics khi không có label

Nếu không có label, notebook vẫn có thể so sánh bằng:

- Silhouette score
- Anomaly distribution
- Anomaly ratio
- PCA visualization

### 9.3 Output so sánh hiện tại

Bảng summary đã tạo:

| Algorithm | Total Users | Anomaly Users | Normal Users | Anomaly Ratio |
| --- | ---: | ---: | ---: | ---: |
| K-Means | 50,000 | 2,500 | 47,500 | 5.00% |
| DBSCAN | 50,000 | 2,816 | 47,184 | 5.63% |
| Isolation Forest | 50,000 | 2,500 | 47,500 | 5.00% |

Output files:

```text
results/csv/algorithm_comparison.csv
results/csv/algorithm_comparison_detailed.csv
```

---

## 10. Kết quả trực quan hóa

Các biểu đồ được lưu trong:

```text
results/figures/
```

Một số figure quan trọng:

| Figure | Mô tả |
| --- | --- |
| `network_graph_sample.png` | Sample social interaction graph |
| `graph_degree_distribution.png` | Phân bố degree |
| `graph_feature_correlation.png` | Tương quan giữa các graph features |
| `isolation_forest_anomaly_score_distribution.png` | Phân bố final anomaly score |
| `pca_isolation_forest_anomaly.png` | PCA visualization cho anomaly của Isolation Forest |
| `isolation_graph_feature_correlation.png` | Tương quan giữa graph features và anomaly scores |
| `compare_algorithms_anomaly_users.png` | Số anomaly users theo thuật toán |
| `compare_algorithms_anomaly_ratio.png` | Tỷ lệ anomaly theo thuật toán |
| `compare_algorithms_metrics.png` | So sánh accuracy, precision, recall và F1 |
| `compare_algorithms_pca_anomaly.png` | So sánh PCA anomaly cho tất cả thuật toán |

---

## 11. Cấu trúc project

Cấu trúc project cũ được giữ nguyên. Phần nâng cấp không yêu cầu tạo kiến trúc `src/` mới.

```text
social-anomaly-detection/
├── assets/
│   ├── pipeline.png
│   └── social_anomaly_clustering_animated_logo.gif
├── configs/
│   └── config.yaml
├── datasets/
│   ├── raw/
│   │   └── bot_detection_data.csv
│   ├── cleaned/
│   │   └── bot_detection_clean.csv
│   └── processed/
│       ├── user_features.csv
│       ├── user_features_scaled.csv
│       ├── user_features_with_label.csv
│       └── user_graph_features.csv
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_dbscan_clustering.ipynb
│   ├── 05_evaluation_visualization_updated.ipynb
│   ├── 06_graph_feature_engineering.ipynb
│   ├── 07_isolation_forest_anomaly_detection.ipynb
│   └── 08_compare_algorithms.ipynb
├── results/
│   ├── csv/
│   ├── figures/
│   └── models/
├── README.md
├── README_VI.md
└── requirements.txt
```

---

## 12. Quy trình chạy notebook

Thứ tự chạy được khuyến nghị:

```text
01_explore_dataset.ipynb
    -> 02_preprocessing_feature_engineering.ipynb
    -> 03_kmeans_clustering.ipynb
    -> 04_dbscan_clustering.ipynb
    -> 05_evaluation_visualization_updated.ipynb
    -> 06_graph_feature_engineering.ipynb
    -> 07_isolation_forest_anomaly_detection.ipynb
    -> 08_compare_algorithms.ipynb
```

Năm notebook đầu là workflow baseline gốc. Ba notebook cuối là phần đóng góp nâng cấp cho đồ án Data Mining.

---

## 13. Cách chạy

### 13.1 Cài dependencies

```bash
pip install -r requirements.txt
```

### 13.2 Mở Jupyter Notebook

```bash
jupyter notebook
```

Sau đó chạy các notebook theo đúng thứ tự được khuyến nghị.

### 13.3 Tùy chọn: chạy bằng nbconvert

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/06_graph_feature_engineering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_isolation_forest_anomaly_detection.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/08_compare_algorithms.ipynb
```

Trên một số môi trường Windows, Jupyter có thể gặp lỗi quyền liên quan đến `SetFileSecurity`. Nếu gặp lỗi đó, chạy:

```powershell
$env:JUPYTER_ALLOW_INSECURE_WRITES='1'
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\06_graph_feature_engineering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\07_isolation_forest_anomaly_detection.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\08_compare_algorithms.ipynb
```

Nên chạy notebook tuần tự. Chạy nhiều notebook song song trên Windows có thể gây xung đột port kernel tạm thời.

---

## 14. Output mong đợi

Sau khi chạy toàn bộ pipeline, các output quan trọng là:

```text
datasets/processed/user_graph_features.csv
results/csv/isolation_forest_results.csv
results/csv/algorithm_comparison.csv
results/csv/algorithm_comparison_detailed.csv
results/models/isolation_forest.pkl
results/figures/
```

Project vẫn giữ các output K-Means và DBSCAN gốc:

```text
results/csv/kmeans_clustered_users.csv
results/csv/kmeans_anomaly_users.csv
results/csv/dbscan_clustered_users.csv
results/csv/dbscan_anomaly_users.csv
results/models/kmeans_model.pkl
results/models/dbscan_model.pkl
```

---

## 15. Ghi chú và hạn chế

- Dataset không có explicit user-to-user edges, vì vậy graph edges được xây dựng bằng các similarity-based proxies.
- `bot_label` chỉ dùng để đánh giá, không dùng để train mô hình unsupervised.
- Isolation Forest được cấu hình như một phương pháp Data Mining đơn giản và dễ giải thích.
- Rule-based suspicious score cố ý được giữ đơn giản và nên được xem là tín hiệu hỗ trợ, không phải classifier độc lập.
- Project không sử dụng Deep Learning, XGBoost, BERT, LSTM hoặc transformer-based text modeling.
- Mục tiêu không phải xây dựng bot detector production, mà là trình bày một anomaly detection pipeline hoàn chỉnh với feature engineering nâng cao.

---

## 16. Kết luận

Phiên bản nâng cấp này biến project clustering cũ thành một đồ án Data Mining hoàn chỉnh hơn. Project giữ K-Means và DBSCAN làm baseline, bổ sung graph feature engineering, huấn luyện Isolation Forest để phát hiện bất thường, và so sánh tất cả phương pháp bằng output và visualization nhất quán.

Project cuối cùng tập trung vào:

- Anomaly detection
- Feature engineering
- Graph-based behavioral features
- Mô hình unsupervised đơn giản
- Evaluation và visualization rõ ràng

Tài liệu tiếng Anh có tại [README.md](./README.md).
