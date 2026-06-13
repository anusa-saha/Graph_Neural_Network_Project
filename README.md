# Financial Fraud Detection using Graph Neural Networks (GATv2)

> **Supervised Node Classification with Temporal Decay-Weighted Graph Attention Networks**
---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [Dataset](#3-dataset)
4. [Prerequisites](#4-prerequisites)
5. [Installation and Setup](#5-installation-and-setup)
6. [Project Architecture](#6-project-architecture)
7. [Methodology](#7-methodology)
   - [Graph Construction](#71-graph-construction)
   - [Feature Engineering](#72-feature-engineering)
   - [Temporal Decay Formulations](#73-temporal-decay-formulations)
   - [Model Architecture (GATv2)](#74-model-architecture-gatv2)
   - [Baseline: MLP](#75-baseline-mlp)
   - [Class Imbalance Strategy](#76-class-imbalance-strategy)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation](#9-evaluation)
10. [Results and Analysis](#10-results-and-analysis)
11. [Repository Structure](#11-repository-structure)
12. [References](#13-references)

---

## 1. Project Overview

This project formulates **credit card and payment fraud detection as a supervised node classification task** using Graph Neural Networks (GNNs). Rather than treating each financial transaction in isolation — the conventional approach of models like XGBoost — we represent transactions as nodes in a graph where edges encode shared account relationships and temporal proximity. This allows the model to learn from the **topological and relational structure** of fraudulent activity.

We propose an enhanced **Graph Attention Network v2 (GATv2)** architecture that fuses multi-headed dynamic attention with three variants of **temporal decay weighting** (Linear, Exponential, and Piecewise/Step). The model is benchmarked against a standard Multi-Layer Perceptron (MLP) baseline to demonstrate the necessity of spatial and temporal graph modeling in isolating sophisticated fraud clusters.

**Key contributions:**

- A transaction graph construction methodology linking nodes via shared `account_id` within a configurable time window.
- Three distinct temporal decay kernels integrated as edge attributes into the GATv2 message-passing framework.
- A cost-sensitive learning strategy using Focal Loss to handle extreme class imbalance (1.71% fraud rate).
- Comprehensive t-SNE embedding visualizations to qualitatively assess class separation learned by each model variant.

---

## 2. Motivation and Problem Statement

### Why Graph Neural Networks for Fraud Detection?

Traditional machine learning pipelines for fraud detection treat each transaction as an independent, tabular data point. This design fundamentally fails to capture the **organized, relational nature** of modern financial fraud, which often manifests as:

- **Fraud rings**: Groups of accounts sharing device IDs, IP addresses, or email addresses to commit coordinated fraud.
- **Velocity patterns**: Rapid sequences of transactions across linked accounts within short time windows.
- **Temporal clustering**: Fraudulent transactions that peak during off-hours (e.g., 00:00–05:00) and exhibit distinct time-delta distributions between related nodes.

By modeling the transaction dataset as a graph `G = (V, E)`, we enable the model to perform **neighborhood aggregation** — learning not just from a transaction's own features, but from the behavioral context of all transactions linked to the same account. The GATv2 attention mechanism then learns to **weight these neighbors dynamically**, prioritizing the most suspicious and temporally relevant connections.

### Research Questions

1. Does incorporating relational graph structure outperform feature-only tabular models for fraud detection?
2. Which temporal decay formulation (Linear, Exponential, or Step) produces the best class separation in learned embeddings?
3. How can cost-sensitive learning be effectively applied to handle a 57:1 class imbalance at inference time?

---

## 3. Dataset

**Source:** [Fraud Detection — 1M Transactions, 7 Fraud Types](https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types) by Sergey Nefedov (Kaggle)

| Property | Detail |
|---|---|
| Size | ~1 Million transactions |
| Time Span | 2022 – 2024 |
| Fraud Rate | **1.71%** (approximately 17,100 fraudulent transactions) |
| Fraud Types | 7 distinct behavioral patterns |
| Generation Method | Probabilistic synthetic model mirroring production environments |

### Fraud Behavioral Signatures

| Pattern | Description |
|---|---|
| **Temporal Skew** | Fraud peaks 2× during the 00:00–05:00 window (night-time concentration) |
| **Topological Clusters** | 200 organized fraud rings sharing Device/IP/Email nodes |
| **Cross-Feature Interaction** | Correlated velocity, amount ratios, and IP risk scores |
| **Class Imbalance** | Strict 1.71% minority class ratio (~57:1 normal-to-fraud) |

### Train / Validation / Test Split

The dataset is split **chronologically by year** to simulate real-world temporal generalization — the model is always predicting the future, never the past.

| Split | Year | Usage |
|---|---|---|
| Train | 2022 | Model parameter optimization |
| Validation | 2023 | Hyperparameter tuning & early stopping |
| Test | 2024 | Final, held-out evaluation |

---

## 4. Prerequisites

A working understanding of the following concepts is recommended before engaging with this codebase:

- **Deep Learning**: Backpropagation, gradient descent, loss functions (BCE, Focal Loss)
- **Graph Theory**: Adjacency representation, node/edge attributes, message passing
- **Attention Mechanisms**: Scaled dot-product attention, multi-head attention
- **GNN Fundamentals**: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT, GATv2)
- **Dimensionality Reduction**: t-SNE for high-dimensional embedding visualization
- **Evaluation Metrics for Imbalanced Datasets**: AUPRC, F1-Score, Precision-Recall curves (accuracy is not informative here)

---

## 5. Installation and Setup

### Environment

Python 3.10+ with a CUDA-enabled GPU is strongly recommended. The graph construction step is memory-intensive for 1M+ transaction graphs.

```bash
# Clone the repository
git clone https://github.com/anusa-saha/Graph_Neural_Network_Project.git
cd Graph_Neural_Network
```

### Core Dependencies

```bash
# Install PyTorch (adjust CUDA version to match your environment)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and sparse ops (must match your PyTorch version)
pip install pyg_lib torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Install remaining dependencies
pip install pandas numpy scikit-learn matplotlib seaborn networkx
```

### Software Stack

| Category | Libraries / Frameworks |
|---|---|
| Core Deep Learning | PyTorch (v2.x), CUDA 11.8+ |
| Graph Framework | PyTorch Geometric (PyG), NetworkX |
| Preprocessing | Scikit-Learn (RobustScaler), Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, t-SNE |
| Metrics | F1-Score, AUPRC, ROC-AUC, Precision/Recall |

### Dataset Download (Kaggle API)

```python
# Configure Kaggle credentials (Colab environment shown)
import os
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'

# Download and extract
os.system('kaggle datasets download -d sergionefedov/fraud-detection-1m-transactions-7-fraud-types')
os.system('unzip fraud-detection-1m-transactions-7-fraud-types.zip')
```

---

## 6. Project Architecture

```
Graph_Neural_Network/
│
├── financial_fraud_detection_using_gnn.ipynb   # Main experiment notebook
├── ai_term_project.pdf                          # Full project report
│
├── models/
│   ├── best_model.pt           # Best GATv2 checkpoint (Exponential decay)
│   ├── best_linear_model.pt    # Best GATv2 checkpoint (Linear decay)
│   └── best_step_model.pt      # Best GATv2 checkpoint (Step decay)
│
└── README.md
```

The entire experimental pipeline is contained in a single, fully documented Jupyter Notebook structured as follows:

1. **Environment Setup** → package installation, GPU configuration
2. **Dataset Ingestion** → Kaggle download, EDA, cleaning
3. **Feature Engineering** → cyclical encoding, log transforms, RobustScaler
4. **Graph Construction** → edge building, temporal weight computation, PyG packaging
5. **Model Definition** → GATv2 (three decay variants) and MLP baseline
6. **Training & Early Stopping** → Focal Loss, AdamW, patience-based stopping
7. **Threshold Optimization** → Precision-Recall curve analysis
8. **t-SNE Visualization** → balanced and imbalanced embedding plots

---

## 7. Methodology

### 7.1 Graph Construction

The dataset is modeled as an **undirected graph** `G = (V, E)` where:

- **Nodes** `V`: Each transaction is a node with a feature vector `x ∈ ℝ^F`.
- **Edges** `E`: An edge `e_ij` is instantiated between transactions `i` and `j` if they share a common `account_id` **and** their absolute time difference `|t_i - t_j|` falls within a configurable temporal window (default: `168 hours = 7 days`).

This construction directly captures **organized fraud ring patterns**, where accounts transacting together are prime candidates for shared fraudulent behavior.

```python
def transaction_graph(df, window=168, lambda_val=0.1):
    # Self-join on account_id to find all transaction pairs
    df_edges = df[['node_id', 'account_id', 'timestamp']].merge(
        df[['node_id', 'account_id', 'timestamp']],
        on='account_id', suffixes=('_src', '_dst')
    )
    # Apply temporal window filter and remove self-loops
    time_diff = (df_edges['timestamp_dst'] - df_edges['timestamp_src']).dt.total_seconds() / 3600.0
    mask = (time_diff.abs() <= window) & (df_edges['node_id_src'] != df_edges['node_id_dst'])
    ...
```

Three separate `PyG Data` objects are constructed per split (train/val/test), one for each decay variant.

### 7.2 Feature Engineering

Raw features undergo the following transformations before being packed as node features:

**Log1p Transformation** (applied before scaling to reduce right-skew):
- `amount`, `credit_limit`, `velocity_1h`, `time_since_last_s`, `amount_vs_avg_ratio`

**RobustScaler** (fit only on the 2022 training split, applied to all splits):
- Robust to outliers; uses the interquartile range. Separate scalers are maintained for each feature family to prevent data leakage.

**Cyclical Encoding** (for time-of-day and day-of-week periodicity):

```python
# Micro-seasonality: hour of day
dataset['hour_sin'] = np.sin(2 * np.pi * dataset['timestamp'].dt.hour / 24.0)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['timestamp'].dt.hour / 24.0)

# Macro-seasonality: month of year
dataset['month_sin'] = np.sin(2 * np.pi * dataset['timestamp'].dt.month / 12.0)
dataset['month_cos'] = np.cos(2 * np.pi * dataset['timestamp'].dt.month / 12.0)
```

**One-Hot Encoding**: Applied to `device_type`, `merchant_category`, and `merchant_country`.

**IP Risk Score**: Normalized to `[0, 1]` by dividing by 100.

### 7.3 Temporal Decay Formulations

A core contribution of this work is the integration of temporal context into the GATv2 attention mechanism via **edge attributes**. Given the time delta between two connected transactions `Δt_ij = |t_i - t_j|` (in hours), three decay kernels are evaluated:

**1. Linear Decay** — enforces a hard cutoff at the temporal window `T`:

$$w_{\text{linear}} = \max\left(0,\ 1 - \frac{\Delta t_{ij}}{T}\right)$$

**2. Exponential Decay** — smooth, continuous decay controlled by rate `λ`:

$$w_{\text{exp}} = e^{-\lambda \cdot \Delta t_{ij}}$$

**3. Piecewise (Step) Decay** — discrete, categorized relevance based on hourly thresholds:

$$w_{\text{step}} = \begin{cases} 1.00 & \text{if } \Delta t_{ij} \leq 24 \\ 0.75 & \text{if } 24 < \Delta t_{ij} \leq 72 \\ 0.50 & \text{if } 72 < \Delta t_{ij} \leq 120 \\ 0.25 & \text{if } \Delta t_{ij} > 120 \end{cases}$$

These weights are passed as scalar edge attributes (`edge_attr`, shape `[num_edges, 1]`) into the GATv2 convolution layers, where they modulate the attention coefficient computation.

### 7.4 Model Architecture (GATv2)

The proposed model addresses the **static attention** limitation of the original GAT (Veličković et al., 2018), where attention scores are decoupled from the query node's representation. GATv2 (Brody et al., 2022) resolves this with a dynamic attention mechanism.

**Fused Temporal Attention Coefficient:**

$$e_{ij} = \vec{a}^T \text{LeakyReLU}\left(\mathbf{W}[h_i \| h_j]\right) + \phi(w_{ij})$$

Where:
- `[· ∥ ·]` denotes vector concatenation.
- `W ∈ ℝ^{d'×2d}` is the learnable linear transformation.
- `ā ∈ ℝ^{d'}` is the dynamic attention vector (per-node ranking, unlike GAT's global ranking).
- `φ(w_ij)` is the temporal bias derived from the decay weight, applied additively to prioritize recent edges.

**Softmax normalization** across each node's neighborhood:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**Layer Summary:**

| Layer | Operation | Input Dim | Output Dim | Notes |
|---|---|---|---|---|
| GATv2Conv 1 | Multi-head attention | `F_in` | `16 × 4 = 64` | 4 heads, `concat=True` |
| ELU | Activation | 64 | 64 | — |
| Dropout | Regularization | 64 | 64 | `p = 0.3` |
| GATv2Conv 2 | Single-head attention | 64 | 16 | 1 head, `concat=False` |
| ELU | Activation | 16 | 16 | — |
| Linear (out) | Projection | 16 | 1 | Logit output |

### 7.5 Baseline: MLP

A standard Multi-Layer Perceptron is trained on the same node features **without any graph structure** (no `edge_index`, no `edge_attr`). This serves as the ablation baseline to quantify the gain from relational modeling.

**FraudMLP Architecture:**

| Layer | Operation | Input Dim | Output Dim | Activation |
|---|---|---|---|---|
| fc1 | Linear | `num_features` | 64 | ELU |
| Dropout | Regularization | 64 | 64 | `p = 0.3` |
| fc2 | Linear | 64 | 16 | ELU |
| out | Linear | 16 | 1 | None (Logit) |

The bottleneck dimension of 16 matches the GATv2 hidden layer size, enabling a fair comparison of 16-dimensional t-SNE embeddings.

### 7.6 Class Imbalance Strategy

With a fraud rate of only 1.71% (~57:1 normal-to-fraud ratio), naive binary cross-entropy loss will cause the model to trivially predict all transactions as normal and achieve 98.29% accuracy while detecting zero fraud.

Two strategies are employed:

**Focal Loss** — down-weights the contribution of easy (well-classified) examples and focuses training on hard negatives. This is critical for the GATv2 which processes all nodes in a single forward pass:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

**Threshold Optimization at Inference** — rather than using the default 0.5 threshold, the optimal decision boundary is determined via the Precision-Recall curve on the validation set:

```python
precision, recall, thresholds = precision_recall_curve(y_true, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
```

This allows the model to trade precision for recall at a calibrated operating point suited to the cost structure of fraud detection.

---

## 8. Training Pipeline

**Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`)

**Early Stopping:** Patience-based stopping monitors validation F1-score. Training halts if no improvement is observed for 20 consecutive epochs. The best model checkpoint is saved to disk.

**Epochs:** Up to 100 (GATv2 variants), up to 150 (MLP baseline).

**Training loop sketch:**

```python
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data_train.x, data_train.edge_index, data_train.edge_attr)
    loss = criterion(out, data_train.y)   # Focal Loss
    loss.backward()
    optimizer.step()

    val_metrics = evaluate(model, data_val)
    early_stopper(val_metrics['f1'])

    if val_metrics['f1'] > best_val_f1:
        torch.save(model.state_dict(), 'best_model.pt')

    if early_stopper.early_stop:
        break
```

Three fully independent training runs are executed — one per temporal decay variant — each with a fresh model initialization and optimizer state.

---

## 9. Evaluation

All final evaluation is performed on the **2024 held-out test set** using the threshold optimized on the validation set.

**Primary Metrics** (appropriate for imbalanced classification):

| Metric | Rationale |
|---|---|
| **F1-Score (Fraud class)** | Harmonic mean of precision and recall; primary optimization target |
| **AUPRC** | Area under the Precision-Recall Curve; preferred over ROC-AUC when classes are heavily imbalanced |
| **ROC-AUC** | Complementary ranking metric |
| **Precision / Recall** | Explicit trade-off characterization |

**Qualitative Evaluation — t-SNE Visualization:**

Intermediate 16-dimensional hidden embeddings (output of GATv2 Layer 2 / MLP fc2) are extracted and projected to 2D using t-SNE. Two visualization regimes are compared for each model:

- **Balanced (50/50)**: 2,500 fraud + 2,500 normal nodes, sampled from the test set.
- **Natural Imbalance**: 10,000 nodes sampled with the real ~57:1 ratio.

Clear spatial separation between the fraud (teal) and normal (grey) clusters in the t-SNE plots serves as qualitative evidence that the model has learned meaningful latent representations of fraudulent behavior.

---

## 10. Results and Analysis

### t-SNE Embedding Comparisons

The t-SNE visualizations reveal a consistent pattern across all GATv2 variants: the model learns to form a **compact, spatially coherent fraud cluster** that is clearly separated from the normal transaction mass. This separation is most pronounced in the exponential decay variant.

| Model Variant | Balanced Separation | Natural Imbalance Separation | Key Observation |
|---|---|---|---|
| **GATv2 + Linear Decay** | Strong | Moderate | Clean boundary; fraud forms a distinct lobe |
| **GATv2 + Exponential Decay** | Strongest | Strongest | Tightest fraud cluster; best inter-class distance |
| **GATv2 + Step Decay** | Strong | Good | Slightly more diffuse than exponential |
| **MLP Baseline** | Weakest | Weakest | Fraud and normal nodes remain significantly intermixed; no clear boundary |

### Key Findings

**Graph structure matters critically.** The MLP baseline, despite having access to the same node features, fails to achieve the class separation of any GATv2 variant. This empirically validates the hypothesis that relational, topological information is essential for fraud ring detection.

**Temporal decay improves representation quality.** All three decay formulations outperform a baseline GATv2 trained without temporal edge attributes. The model learns to discount stale transaction history and focus attention on recent, behaviorally similar neighbors.

**Exponential decay is the most effective kernel.** Its smooth, continuous nature provides the most informative gradient signal for the attention mechanism, enabling tighter within-class clustering in the latent space.

**Extreme class imbalance remains a persistent challenge.** Even with Focal Loss and threshold optimization, the natural 57:1 imbalance in the test set makes achieving simultaneously high precision and high recall difficult. This is an open problem inherent to production fraud datasets.

---

## 11. Repository Structure

```
.
├── financial_fraud_detection_using_gnn.ipynb
│   ├── Section 1:  Package Installation & GPU Setup
│   ├── Section 2:  Dataset Download (Kaggle API)
│   ├── Section 3:  EDA & Statistical Analysis
│   ├── Section 4:  Feature Engineering
│   │               ├── Log1p Transforms
│   │               ├── RobustScaler (fit on train only)
│   │               └── Cyclical Time Encoding
│   ├── Section 5:  Graph Construction
│   │               ├── Self-join on account_id
│   │               ├── Temporal window filter (168h)
│   │               ├── Linear / Exponential / Step decay edge attrs
│   │               └── PyG Data packaging
│   ├── Section 6:  t-SNE (Raw Data Baseline)
│   ├── Section 7:  GATv2 Model Definition
│   ├── Section 8:  Cost-Sensitive Loss & Class Weighting
│   ├── Section 9:  Training — Exponential Decay GATv2
│   ├── Section 10: Threshold Optimization (Precision-Recall Curve)
│   ├── Section 11: Training — Linear Decay GATv2
│   ├── Section 12: Training — Step Decay GATv2
│   ├── Section 13: MLP Baseline Training
│   └── Section 14: t-SNE Visualization (All Models — Balanced & Imbalanced)
│
└── project_report.pdf   # Full project report
```

---


## 12. References

[1] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph Attention Networks," *International Conference on Learning Representations (ICLR)*, 2018.

[2] S. Brody, U. Alon, and E. Yahav, "How Attentive are Graph Attention Networks?" *International Conference on Learning Representations (ICLR)*, 2022.

[3] M. Fey and J. E. Lenssen, "Fast Graph Representation Learning with PyTorch Geometric," *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019.

[4] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.

[5] D. Cheng, Y. Zou, S. Xiang, and C. Jiang, "Graph Neural Networks for Financial Fraud Detection: A Review," *arXiv preprint arXiv:2411.05815*, 2024.

[6] S. Nefedov, "Fraud Detection — 1M Transactions, 7 Fraud Types," Kaggle Dataset, 2024. Available: https://www.kaggle.com/datasets/sergionefedov/fraud-detection-1m-transactions-7-fraud-types

---

*For questions, issues, or collaboration inquiries, please open a GitHub Issue or reach out via the repository.*
