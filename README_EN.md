<h1 align="center">Social Media Anomaly Account Detection</h1>
<h3 align="center">Isolation Forest with Graph Features</h3>

<p align="center">
  <b>Data Mining Project | K-Means | Graph Feature Engineering | Isolation Forest</b>
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
  <b><a href="./README.md">Vietnamese Version</a></b>
</div>

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Project Goals](#2-project-goals)
- [3. Dataset](#3-dataset)
- [4. Methodology](#4-methodology)
- [5. Baseline Clustering](#5-baseline-clustering)
- [6. Graph Feature Engineering](#6-graph-feature-engineering)
- [7. Isolation Forest Anomaly Detection](#7-isolation-forest-anomaly-detection)
- [8. Rule-Based Suspicious Score](#8-rule-based-suspicious-score)
- [9. Algorithm Comparison](#9-algorithm-comparison)
- [10. Visualization Outputs](#10-visualization-outputs)
- [11. Project Structure](#11-project-structure)
- [12. Notebook Workflow](#12-notebook-workflow)
- [13. How to Run](#13-how-to-run)
- [14. Expected Outputs](#14-expected-outputs)
- [15. Notes and Limitations](#15-notes-and-limitations)
- [16. Conclusion](#16-conclusion)

---

## 1. Project Overview

This project detects anomalous social media accounts using a Data Mining pipeline. It is upgraded from an older clustering-based project that used **K-Means** to find abnormal user behavior.

The new project topic is:

```text
Social media anomaly account detection using Isolation Forest with graph features
```

The upgrade keeps the baseline project structure, baseline notebooks, preprocessing code, feature engineering steps, scaling, PCA visualization, and clustering baseline. The new contribution is an additional anomaly detection pipeline based on:

- Graph feature engineering
- Isolation Forest
- Simple rule-based suspicious scoring
- Comparison with K-Means

---

## 2. Project Goals

The main goals are:

- Preserve the original clustering project (K-Means) and use it as a baseline.
- Reuse the existing data loading, preprocessing, feature engineering, scaling, PCA, evaluation, and visualization logic.
- Build a user interaction graph from the available social media metadata.
- Extract graph-based features for each user.
- Train an Isolation Forest model for unsupervised anomaly detection.
- Add a simple rule-based suspicious score to make the anomaly score more interpretable.
- Compare K-Means and Isolation Forest in one consistent pipeline.
- Produce CSV outputs, trained models, and visualizations for a complete Data Mining report.

---

## 3. Dataset

The project uses a Twitter/X-style bot detection dataset.

Main raw file:

```text
datasets/raw/bot_detection_data.csv
```

Processed files from the pipeline:

```text
datasets/cleaned/bot_detection_clean.csv
datasets/processed/user_features.csv
datasets/processed/user_features_scaled.csv
datasets/processed/user_features_with_label.csv
```

### 3.1 Main Columns

| Column | Description | Usage |
| --- | --- | --- |
| `user_id` | Unique user identifier | Node ID and result mapping |
| `username` | Account username | Used to derive username length |
| `tweet` | Tweet content | Used to derive tweet length |
| `retweet_count` | Number of retweets | Behavior feature |
| `mention_count` | Number of mentions | Behavior feature |
| `follower_count` | Number of followers | Behavior feature |
| `verified` | Account verification status | Behavior feature |
| `hashtags` | Hashtag/token text | Hashtag similarity and hashtag features |
| `created_at` | Date/time information | Account age feature |
| `bot_label` | Available label | Evaluation only, not used for unsupervised training |

The label column is used only after model training to evaluate the results. The unsupervised models do not use `bot_label` during fitting.

---

## 4. Methodology

The upgraded project follows this Data Mining pipeline:

```text
Raw dataset
    -> Data cleaning
    -> Feature engineering
    -> Feature scaling
    -> Baseline clustering with K-Means
    -> Graph construction
    -> Graph feature extraction
    -> Isolation Forest anomaly detection
    -> Rule-based suspicious score
    -> Algorithm comparison
    -> Visualization and reporting
```

The baseline clustering workflow is preserved so the new method can be compared against the original project.

---

## 5. Baseline Clustering

The original project notebooks are kept:

```text
notebooks/01_explore_dataset.ipynb
notebooks/02_preprocessing_feature_engineering.ipynb
notebooks/03_kmeans_clustering.ipynb
notebooks/05_evaluation_visualization_updated.ipynb
```

### K-Means

K-Means groups users into behavior clusters. Users far from cluster centers are treated as potential anomalies (top 5% of anomaly scores).

Main outputs:

```text
results/csv/kmeans_clustered_users.csv
results/csv/kmeans_anomaly_users.csv
results/models/kmeans_model.pkl
```

This algorithm remains important because it provides the baseline for evaluating whether the new graph-enhanced Isolation Forest pipeline adds value.

---

## 6. Graph Feature Engineering

New notebook:

```text
notebooks/06_graph_feature_engineering.ipynb
```

Each user is represented as a graph node.

The dataset does not contain direct user-to-user relations such as `user A mentions user B` or `user A retweets user B`. Because of that, the graph is built from relation proxies available in the dataset:

- `feature_similarity`: cosine similarity between scaled user behavior features.
- `hashtag_similarity`: users sharing similar hashtag/token text.
- `interaction_similarity`: users with similar `mention_count`, `retweet_count`, and `hashtag_count`.

### 6.1 Generated Graph Features

The notebook computes the following graph features for each user:

| Feature | Meaning |
| --- | --- |
| `degree` | Number of graph connections for a user |
| `degree_centrality` | Relative node centrality based on degree |
| `clustering_coefficient` | How strongly the user belongs to local graph neighborhoods |
| `interaction_score` | Normalized weighted degree / interaction strength |
| `pagerank` | Graph importance score based on link structure |

Output:

```text
datasets/processed/user_graph_features.csv
```

---

## 7. Isolation Forest Anomaly Detection

New notebook:

```text
notebooks/07_isolation_forest_anomaly_detection.ipynb
```

Isolation Forest is used as the main upgraded anomaly detection algorithm. It is suitable for this project because:

- It is unsupervised.
- It works well with numerical behavior features.
- It detects observations that are easier to isolate from the majority.
- It is simpler and more appropriate for a Data Mining project than Deep Learning models.

### 7.1 Feature Set

Isolation Forest uses both old behavior features and new graph features.

Old behavior features:

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

New graph features:

```text
degree
degree_centrality
clustering_coefficient
interaction_score
pagerank
```

### 7.2 Output Columns

The Isolation Forest result file contains:

| Column | Meaning |
| --- | --- |
| `isolation_score` | Normalized Isolation Forest anomaly score |
| `anomaly_score` | Alias for the Isolation Forest score |
| `anomaly_label` | Isolation Forest label, `-1 = anomaly`, `1 = normal` |
| `suspicious_score` | Rule-based suspicious behavior score |
| `final_anomaly_score` | Combined anomaly score |
| `final_anomaly_label` | Final label, `-1 = anomaly`, `1 = normal` |
| `is_anomaly` | Binary label for comparison, `1 = anomaly`, `0 = normal` |

Main outputs:

```text
results/csv/isolation_forest_results.csv
results/models/isolation_forest.pkl
```

---

## 8. Rule-Based Suspicious Score

The project adds a simple suspicious score to make the final anomaly score more interpretable.

This score is not a complex machine learning model. It is based on straightforward behavior rules such as:

- Very high hashtag count.
- Very high hashtag ratio.
- Very high mention count.
- Very high retweet count.
- High retweet activity compared with follower count.
- Low followers but high activity.
- Very high graph interaction score.

Each triggered condition adds one suspicious point. The score is then normalized to `[0, 1]`.

Final score:

```text
final_anomaly_score = 0.7 * isolation_score + 0.3 * suspicious_score
```

This keeps the model simple while still adding domain-oriented behavior signals.

---

## 9. Algorithm Comparison

New notebook:

```text
notebooks/08_compare_algorithms.ipynb
```

This notebook compares:

- K-Means (Baseline)
- Isolation Forest with graph features (Upgraded)

### 9.1 Metrics When Labels Are Available

If `bot_label` exists, the notebook computes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 9.2 Metrics Without Labels

If labels are unavailable, the notebook can still compare:

- Silhouette score
- Anomaly distribution
- Anomaly ratio
- PCA visualization

### 9.3 Current Comparison Output

The generated comparison summary:

| Algorithm | Total Users | Anomaly Users | Normal Users | Anomaly Ratio |
| --- | ---: | ---: | ---: | ---: |
| K-Means | 50,000 | 2,500 | 47,500 | 5.00% |
| Isolation Forest | 50,000 | 2,500 | 47,500 | 5.00% |

Output files:

```text
results/csv/algorithm_comparison.csv
results/csv/algorithm_comparison_detailed.csv
```

---

## 10. Visualization Outputs

Visualizations are saved in:

```text
results/figures/
```

Important figures include:

| Figure | Description |
| --- | --- |
| `network_graph_sample.png` | Sample social interaction graph |
| `graph_degree_distribution.png` | Degree distribution |
| `graph_feature_correlation.png` | Correlation between graph features |
| `isolation_forest_anomaly_score_distribution.png` | Final anomaly score distribution |
| `pca_isolation_forest_anomaly.png` | PCA visualization for Isolation Forest anomalies |
| `isolation_graph_feature_correlation.png` | Correlation between graph features and anomaly scores |
| `compare_algorithms_anomaly_users.png` | Anomaly user count by algorithm |
| `compare_algorithms_anomaly_ratio.png` | Anomaly ratio by algorithm |
| `compare_algorithms_metrics.png` | Accuracy, precision, recall, and F1 comparison |
| `compare_algorithms_pca_anomaly.png` | PCA anomaly comparison for algorithms |

---

## 11. Project Structure

The project structure is as follows:

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
│   ├── 05_evaluation_visualization_updated.ipynb
│   ├── 06_graph_feature_engineering.ipynb
│   ├── 07_isolation_forest_anomaly_detection.ipynb
│   └── 08_compare_algorithms.ipynb
├── results/
│   ├── csv/
│   │   ├── algorithm_comparison.csv
│   │   ├── algorithm_comparison_detailed.csv
│   │   ├── anomaly_summary.csv
│   │   ├── cluster_summary.csv
│   │   ├── evaluation_summary.csv
│   │   ├── feature_summary_by_anomaly.csv
│   │   ├── isolation_forest_results.csv
│   │   └── kmeans_anomaly_users.csv
│   ├── figures/
│   └── models/
│       ├── isolation_forest.pkl
│       ├── kmeans_model.pkl
│       ├── pca_model.pkl
│       └── scaler.pkl
├── README.md
├── README_VI.md
└── requirements.txt
```

---

## 12. Notebook Workflow

Recommended execution order:

```text
01_explore_dataset.ipynb
    -> 02_preprocessing_feature_engineering.ipynb
    -> 03_kmeans_clustering.ipynb
    -> 05_evaluation_visualization_updated.ipynb
    -> 06_graph_feature_engineering.ipynb
    -> 07_isolation_forest_anomaly_detection.ipynb
    -> 08_compare_algorithms.ipynb
```

The first four notebooks are the baseline workflow. The last three notebooks are the upgraded Data Mining contribution.

---

## 13. How to Run

### 13.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 13.2 Start Jupyter Notebook

```bash
jupyter notebook
```

Then run the notebooks in the recommended order.

### 13.3 Optional: Execute with nbconvert

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/06_graph_feature_engineering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/07_isolation_forest_anomaly_detection.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/08_compare_algorithms.ipynb
```

On some Windows environments, Jupyter may fail with a permission error related to `SetFileSecurity`. If that happens, run:

```powershell
$env:JUPYTER_ALLOW_INSECURE_WRITES='1'
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\06_graph_feature_engineering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\07_isolation_forest_anomaly_detection.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks\08_compare_algorithms.ipynb
```

Run the notebooks sequentially. Running multiple notebooks in parallel can cause temporary Jupyter kernel port conflicts on Windows.

---

## 14. Expected Outputs

After running the full pipeline, the expected key outputs are:

```text
datasets/processed/user_graph_features.csv
results/csv/isolation_forest_results.csv
results/csv/algorithm_comparison.csv
results/csv/algorithm_comparison_detailed.csv
results/models/isolation_forest.pkl
results/figures/
```

The project also keeps the original K-Means baseline outputs:

```text
results/csv/kmeans_clustered_users.csv
results/csv/kmeans_anomaly_users.csv
results/models/kmeans_model.pkl
```

---

## 15. Notes and Limitations

- The dataset does not contain explicit user-to-user edges, so graph edges are constructed from similarity-based proxies.
- `bot_label` is used only for evaluation, not for unsupervised model training.
- Isolation Forest is configured as an unsupervised, interpretable Data Mining method.
- The rule-based suspicious score is intentionally simple and should be treated as a supporting signal, not a standalone classifier.
- The project does not use Deep Learning, XGBoost, BERT, LSTM, or transformer-based text modeling.
- The goal is not to build a production bot detector, but to demonstrate a complete anomaly detection pipeline with enhanced graph feature engineering.

---

## 16. Conclusion

This upgraded version turns the old clustering project into a more complete Data Mining project. It keeps K-Means as a baseline, adds graph feature engineering, trains Isolation Forest for anomaly detection, and compares both methods with consistent outputs and visualizations.

The final project focuses on:

- Anomaly detection
- Feature engineering
- Graph-based behavioral features
- Simple unsupervised modeling
- Clear evaluation and visualization

The Vietnamese documentation is available in [README_VI.md](./README_VI.md).
v