# LysTriFuse

**LysTriFuse: ESM2 and K-mer Decision-Level Fusion with Hybrid Resampling and Triple-Classifier Ensemble for Multiple Lysine Modification Site Prediction**

---

## Overview

LysTriFuse is a computational tool for the simultaneous prediction of four types of lysine post-translational modification (PTM) sites in human proteins: **Acetylation (A)**, **Crotonylation (C)**, **Methylation (M)**, and **Succinylation (S)**.

The key challenges addressed by LysTriFuse are:

- **Extreme class imbalance**: the training set has a max-to-min class ratio exceeding 100:1
- **Multi-label complexity**: co-occurrence and crosstalk between different modification types
- **Limited feature expressiveness** of traditional hand-crafted features

LysTriFuse tackles these challenges through a three-stage pipeline: dual-view feature extraction, a four-stage hybrid resampling strategy, and a weighted heterogeneous classifier ensemble with decision-level fusion.

---

## Framework Overview

```
Input: 49-aa peptide sequence centered on Lysine (K)
         │
         ├─── ESM2 Feature Extraction (100-dim)
         │         └── Linear projection + mean pooling
         │
         └─── K-mer Feature Extraction (20-dim, k=1)
                   └── Amino acid count vector
         │
         ▼
Hybrid Resampling (applied independently to each feature stream)
   Step 1: KPCA Oversampling       (minority class augmentation)
   Step 2: ENN Undersampling × 5   (boundary denoising)
   Step 3: OSS Undersampling × 5   (boundary refinement)
   Step 4: CC Undersampling × 1    (cluster-centroid compression)
         │
         ▼
Triple-Classifier Ensemble (each trained on both feature streams independently)
   ┌─────────────┐   ┌───────────┐   ┌───────────┐
   │  LinearSVC  │   │   SAPP    │   │   FNet    │
   │  w = 0.6    │   │  w = 0.1  │   │  w = 0.3  │
   └─────────────┘   └───────────┘   └───────────┘
         │                 │               │
    avg(ESM2, Kmer)   avg(ESM2, Kmer)  avg(ESM2, Kmer)
         │
         ▼
   Weighted Fusion → Binary prediction (threshold τ = 0.5)
         │
         ▼
Output: 4-bit multi-label vector [A, C, M, S]
```

---

## Dataset

Peptide sequences are derived from the [CPLM 4.0 database](https://cplm.biocuckoo.cn). Lysine-centered 49-residue windows were extracted, deduplicated (100% identity), and split 70/30 into training and test sets. CD-HIT (threshold 0.4) was applied to remove homologous sequences across splits.

| Class | Modification | One-hot | Train | Test |
|-------|-------------|---------|-------|------|
| 0 | A (Acetylation) | [1,0,0,0] | 9,279 | 4,062 |
| 1 | C (Crotonylation) | [0,1,0,0] | 710 | 304 |
| 2 | M (Methylation) | [0,0,1,0] | 600 | 257 |
| 3 | S (Succinylation) | [0,0,0,1] | 454 | 194 |
| 4 | A,C | [1,1,0,0] | 561 | 240 |
| 5 | A,M | [1,0,1,0] | 252 | 107 |
| 6 | A,S | [1,0,0,1] | 360 | 154 |
| 7 | C,M | [0,1,1,0] | 88 | 42 |
| 8 | A,C,M | [1,1,1,0] | 153 | 73 |
| 9 | A,C,S | [1,1,0,1] | 454 | 191 |
| 10 | A,C,M,S | [1,1,1,1] | 73 | 36 |
| **Total** | | | **12,984** | **5,660** |

---

## Methods

### 1. Feature Extraction

**ESM2 (100-dim)**
- Model: `facebook/esm2_t6_8M_UR50D`
- A trainable linear projection layer reduces each position's 320-dim hidden state to 100 dims
- Mean pooling over all positions yields a fixed-size global sequence embedding

**K-mer (20-dim, k=1)**
- Raw amino acid count vector for all 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY)
- Biologically interpretable; directly reflects local amino acid composition preferences

**Fusion strategy**: prediction-level fusion (not feature concatenation). Each classifier is trained independently on ESM2 features and K-mer features; the two prediction probability vectors are averaged as the classifier's combined output. This avoids the curse of dimensionality and preserves each feature stream's discriminative structure.

### 2. Hybrid Resampling

To address extreme imbalance (100:1 ratio), a four-stage strategy is applied sequentially:

| Step | Method | Key Parameters | Rounds |
|------|--------|---------------|--------|
| 1 | KPCA Oversampling | kernel=RBF, γ=0.1, n_components=3, k=5 | 1 |
| 2 | ENN Undersampling | k=3 | 5 |
| 3 | OSS Undersampling | n_neighbors=3 | 5 |
| 4 | CC Undersampling | MiniBatchKMeans, n_init=10, batch_size=2048 | 1 |

Target: **1,180 samples per class**.

### 3. Classifier Ensemble

Three heterogeneous classifiers are combined via weighted averaging:

- **LinearSVC** (w=0.6): strong multi-label coverage in high-dimensional spaces; decision scores transformed to probabilities via sigmoid
- **SAPP** (w=0.1): structure-aware attention mechanism; cross-attention between sequence and structural features; AdamW optimizer, BCE loss
- **FNet** (w=0.3): replaces self-attention with 2D FFT for frequency-domain feature mixing; lightweight and complementary

Final prediction: weighted average of the three classifiers' outputs, binarized at threshold τ=0.5.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Aiming** | Precision averaged over samples |
| **Coverage** | Recall averaged over samples |
| **Accuracy** | Jaccard similarity averaged over samples |
| **Absolute True** | Fraction of samples with all labels correctly predicted |
| **Absolute False** | Fraction of incorrectly predicted label pairs (lower is better) |
| **MRj** | Match Rate: fraction of samples with ≥j true modifications where ≥j are correctly predicted |

MR3 is the primary metric: it measures the model's ability to identify samples with **three or more co-occurring modifications**.

---

## Code Structure

```
LysTriFuse/
├── Readme.md
├── Data/
│   ├── Train dataset/
│   │   ├── (1)Train9279.txt                    # A only
│   │   ├── (2)Train710.txt                     # C only
│   │   ├── (3)Train600.txt                     # M only
│   │   ├── (4)Train454.txt                     # S only
│   │   ├── (5)Train561.txt                     # A,C
│   │   ├── (6)Train251.txt                     # A,M
│   │   ├── (7)Train360.txt                     # A,S
│   │   ├── (8)Train88.txt                      # C,M
│   │   ├── (9)Train153.txt                     # A,C,M
│   │   ├── (10)Train454.txt                    # A,C,S
│   │   └── (11)Train73.txt                     # A,C,M,S
│   └── Test dataset/
│       ├── (1)Test4062.txt                     # A only
│       ├── (2)Test304.txt                      # C only
│       ├── (3)Test257.txt                      # M only
│       ├── (4)Test194.txt                      # S only
│       ├── (5)Test240.txt                      # A,C
│       ├── (6)Test107.txt                      # A,M
│       ├── (7)Test154.txt                      # A,S
│       ├── (8)Test42.txt                       # C,M
│       ├── (9)Test72.txt                       # A,C,M
│       ├── (10)Test194.txt                     # A,C,S
│       └── (11)Test36.txt                      # A,C,M,S
└── Code/
    ├── feature_extraction/
    │   ├── esm2_feature_extraction.py          # ESM2 deep semantic feature extraction
    │   └── kmer_feature_extraction.py          # K-mer amino acid count feature extraction
    ├── resampling/
    │   ├── KPCA_OVER_sample.py                 # Step 1: KPCA oversampling
    │   ├── ENN_UNDER_sample.py                 # Step 2: ENN undersampling
    │   ├── OSS_UNDER_sample.py                 # Step 3: OSS undersampling
    │   └── CC_UNDER_sample.py                  # Step 4: Cluster Centroids undersampling
    └── classification/
        ├── classify_SVM_SAPP_FNet_5-fold_cross-validation_decision_fusion.py
        └── classify_SVM_SAPP_FNet_independent_test_decision_fusion.py
```

---

## Environment

```
Python 3.x
PyTorch 2.7.1+cu128
scikit-learn 1.3.2
transformers
imbalanced-learn
numpy
chardet
```

Hardware used during development: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), Intel Core i7-14700HX, 47.7 GB RAM.

---

## Installation

```bash
git clone https://github.com/bmz24h2/LysTriFuse.git
cd LysTriFuse
pip install -r requirements.txt
```

---

## Usage

> **Note**: Before running any script, open it and update the paths to match your own environment.

### Step 1: Feature Extraction

```bash
python Code/feature_extraction/esm2_feature_extraction.py
python Code/feature_extraction/kmer_feature_extraction.py
```

### Step 2: Hybrid Resampling

Run the four scripts in the following order. **ENN and OSS are each run once per execution — the 5 rounds described in the paper are achieved by manually repeating each script 5 times in sequence**, passing the output of one run as the input of the next.

```bash
# Step 1: KPCA oversampling (run once)
python Code/resampling/KPCA_OVER_sample.py

# Step 2: ENN undersampling (run 5 times manually, chaining output → input each time)
python Code/resampling/ENN_UNDER_sample.py  # repeat 5×

# Step 3: OSS undersampling (run 5 times manually, chaining output → input each time)
python Code/resampling/OSS_UNDER_sample.py  # repeat 5×

# Step 4: CC undersampling (run once)
python Code/resampling/CC_UNDER_sample.py
```

### Step 3: Training & Evaluation

```bash
# 5-fold cross-validation
python Code/classification/classify_SVM_SAPP_FNet_5-fold_cross-validation_decision_fusion.py

# Independent test set
python Code/classification/classify_SVM_SAPP_FNet_independent_test_decision_fusion.py
```

---

## Data Format

Input files should be FASTA-format `.txt` files placed in `Train dataset/` and `Test dataset/` directories under `Data/`. Each sequence should be a 49-residue lysine-centered peptide window.

Feature files and resampled datasets are saved as `.npz` files containing:
- `embeddings`: feature matrix of shape `(n_samples, n_features)`
- `names`: sample identifier strings

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

- Yun Zuo: zuoyun@jiangnan.edu.cn
- JiaYi Ji: 1191230307@stu.jiangnan.edu.cn
- School of Artificial Intelligence and Computer Science, Jiangnan University, Wuxi 214000, China
