import numpy as np
import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, TensorDataset
import math


# Configuration section (all tunable parameters centralized here)

# Random seed
RANDOM_SEED = 42

# Ensemble weights
FIXED_WEIGHTS = {
    'SVM':  0.6,
    'SAPP': 0.1,
    'FNet': 0.3,
}

# Decision threshold
THRESHOLD = 0.5        # Probability above this value is classified as positive; options: 0.3 / 0.4 / 0.5

# Data paths
TRAIN_DIR_FEATURE1 = "Test_22_KPCA_CC_OSS_ENN_OVER_UNDER_balanced_dataset_22"
TRAIN_DIR_FEATURE2 = "DACC_data1/DACC_KPCA1_ENN3_OSS_CC_Train35_dataset"
TEST_DIR_FEATURE1  = "ESM2_FE_Test_dataset_max_512"
TEST_DIR_FEATURE2  = "DACC_data1/DACC_FE_Test_dataset"

# Data classes
NUM_CLASSES = 11

# SVM hyperparameters
SVM_C_CANDIDATES   = [0.1, 1, 10]
SVM_GRIDSEARCH_CV  = 3
SVM_MAX_ITER       = 10000
SVM_TOL            = 1e-4

# SAPP hyperparameters
SAPP_HIDDEN_DIM       = 256
SAPP_N_LAYERS         = 2
SAPP_ATTN_HEADS       = 4
SAPP_FEED_FORWARD_DIM = 758
SAPP_DROPOUT          = 0.2
SAPP_LR               = 0.0005
SAPP_WEIGHT_DECAY     = 1e-6
SAPP_EPOCHS           = 50
SAPP_BATCH_SIZE_TRAIN = 64
SAPP_BATCH_SIZE_TEST  = 128

# FNet hyperparameters
FNET_HIDDEN_DIM  = 512
FNET_NUM_LAYERS  = 2
FNET_DROPOUT     = 0.1
FNET_LR          = 0.001
FNET_EPOCHS      = 50
FNET_BATCH_SIZE  = 64


# Label definitions
class_labels = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
]

# Set random seeds for reproducibility (enhanced version)
def set_seed(seed=42):
    """Set all random seeds to ensure reproducible results"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """DataLoader worker initialization function"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# FNet model implementation
class FNetLayer(nn.Module):
    """FNet Layer with Fourier Transform"""
    def __init__(self, hidden_dim, dropout=FNET_DROPOUT):
        super(FNetLayer, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Fourier transform mixing
        fourier_x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = x + fourier_x
        x = self.norm1(x)
        # Feed-forward network
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class FNetClassifier(nn.Module):
    """FNet - replaces attention mechanism with Fourier transform"""

    def __init__(self, input_dim, hidden_dim=FNET_HIDDEN_DIM, num_layers=FNET_NUM_LAYERS, dropout=FNET_DROPOUT):
        super(FNetClassifier, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # FNet layers
        self.fnet_layers = nn.ModuleList([
            FNetLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        # Project and add sequence dimension
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        # FNet layers
        for fnet_layer in self.fnet_layers:
            x = fnet_layer(x)
        # Pool and classify
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        return self.classifier(x)


class FNetBinaryClassifier:
    """FNet binary classifier wrapper"""
    def __init__(self, input_dim, hidden_dim=FNET_HIDDEN_DIM, num_layers=FNET_NUM_LAYERS, dropout=FNET_DROPOUT,
                 lr=FNET_LR, epochs=FNET_EPOCHS, batch_size=FNET_BATCH_SIZE, random_state=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        # Create model
        self.model = FNetClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"      Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class MultiOutputFNet:
    """Multi-output wrapper for FNet - handles multi-label classification"""

    def __init__(self, input_dim, num_outputs=4, hidden_dim=FNET_HIDDEN_DIM, num_layers=FNET_NUM_LAYERS,
                 dropout=FNET_DROPOUT, lr=FNET_LR, epochs=FNET_EPOCHS, batch_size=FNET_BATCH_SIZE, random_state=None):
        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.classifiers = []

    def fit(self, X, y):
        """Fit a separate FNet model for each output"""
        self.classifiers = []
        for i in range(self.num_outputs):
            print(f"    Training FNet for output {i + 1}/{self.num_outputs}...")
            clf = FNetBinaryClassifier(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                lr=self.lr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                random_state=self.random_state
            )
            clf.fit(X, y[:, i].astype(int))
            self.classifiers.append(clf)
        return self

    def predict(self, X):
        """Predict all outputs"""
        predictions = []
        for clf in self.classifiers:
            pred = clf.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)

    def predict_proba(self, X):
        """Get probability predictions for all outputs (positive class probability)"""
        probas = []
        for clf in self.classifiers:
            proba = clf.predict_proba(X)[:, 1]  # Get positive class probability
            probas.append(proba)
        return np.column_stack(probas)


# SAPP model components
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadAdjAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        assert (hidden_dim % n_heads == 0)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // self.n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.str_fc_k = nn.Linear(hidden_dim, 128)
        self.str_fc_v = nn.Linear(hidden_dim, 128)
        self.fc_o = nn.Linear(hidden_dim + 128, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.str_scale = torch.sqrt(torch.FloatTensor([1])).to(device)

    def forward(self, query, key, value, x_rsa, mask=None, rsa_mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        str_K = self.str_fc_k(key)
        str_V = self.str_fc_v(value)
        rsa_Q = x_rsa.reshape(batch_size, -1, 1, 128).permute(0, 2, 1, 3)
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        str_K = str_K.reshape(batch_size, -1, 1, 128).permute(0, 2, 1, 3)
        str_V = str_V.reshape(batch_size, -1, 1, 128).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))
        rsa_energy = torch.matmul(rsa_Q, str_K.permute(0, 1, 3, 2))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            rsa_energy = rsa_energy.masked_fill(mask == 0, -1e10)
        if rsa_mask is not None:
            rsa_energy = rsa_energy.masked_fill(rsa_mask == 0, -1e10)
        energy = energy / self.scale
        rsa_energy = rsa_energy / self.str_scale
        attention = torch.softmax(energy, dim=-1)
        rsa_attention = torch.softmax(rsa_energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        rsa_x = torch.matmul(self.dropout(rsa_attention), str_V)
        x = x.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.hidden_dim)
        rsa_x = rsa_x.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, 128)
        x = torch.concat((x, rsa_x), dim=-1)
        x = self.fc_o(x)
        return x, [attention, rsa_attention]


class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio, output_dim=None):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        if output_dim is not None:
            self.fc_2 = nn.Linear(pf_dim, output_dim)
        else:
            self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(self.gelu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, attn_heads, feed_forward_hidden, dropout, device, attn_dropout=0.2):
        super().__init__()
        self.self_attention_layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward_layer_norm = nn.LayerNorm(hidden_dim)
        self.adj_attention = MultiHeadAdjAttentionLayer(hidden_dim, attn_heads, attn_dropout, device)
        self.adj_attention_layer_norm = nn.LayerNorm(hidden_dim)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(hidden_dim, feed_forward_hidden, attn_dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, x_rsa, mask, rsa_mask):
        _x, attns = self.adj_attention(x, x, x, x_rsa, mask=mask, rsa_mask=rsa_mask)
        x = self.adj_attention_layer_norm(x + self.dropout(_x))
        _x = self.positionwise_feedforward(x)
        x = self.feedforward_layer_norm(x + self.dropout(_x))
        return x, attns


class SAPPMultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, num_labels=4, hidden_dim=SAPP_HIDDEN_DIM, n_layers=SAPP_N_LAYERS,
                 attn_heads=SAPP_ATTN_HEADS, feed_forward_dim=SAPP_FEED_FORWARD_DIM, dropout=SAPP_DROPOUT, device='cuda'):
        super().__init__()
        self.device = device
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_embedding = PositionalEmbedding(d_model=hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, attn_heads, feed_forward_dim, dropout, device)
            for _ in range(n_layers)
        ])
        self.rsa_replacement = nn.Linear(hidden_dim, 128)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)
        x = x + self.positional_embedding(x)
        x_rsa = self.rsa_replacement(x)

        for transformer in self.transformer_blocks:
            x, _ = transformer(x, x_rsa, None, None)

        x = x.mean(dim=1)
        return torch.sigmoid(self.classifier(x))

# SVM model components
def create_optimized_svm(X_train, y_train):
    """Create and optimize SVM classifier"""
    base_svm = LinearSVC(
        dual=False,
        random_state=RANDOM_SEED,
        max_iter=SVM_MAX_ITER,
        tol=SVM_TOL
    )

    ovr_svm = OneVsRestClassifier(base_svm)

    param_grid = [
        {
            'estimator__C': SVM_C_CANDIDATES,
            'estimator__penalty': ['l1'],
            'estimator__loss': ['squared_hinge']
        },
        {
            'estimator__C': SVM_C_CANDIDATES,
            'estimator__penalty': ['l2'],
            'estimator__loss': ['squared_hinge']
        }
    ]

    grid_search = GridSearchCV(
        estimator=ovr_svm,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=SVM_GRIDSEARCH_CV,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    print(f"  SVM best params: {grid_search.best_params_}")
    print(f"  SVM best score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

# Metric calculation functions
def calculate_custom_metrics(labels_test, y_pred_test, max_tags=4):
    metrics = []
    true_counts = np.sum(labels_test == 1, axis=1)
    match_counts = np.sum((labels_test == 1) & (y_pred_test == 1), axis=1)
    for j in range(1, max_tags + 1):
        valid_samples = true_counts >= j
        if np.sum(valid_samples) > 0:
            correct_samples = (match_counts >= j) & valid_samples
            mr_j = np.sum(correct_samples) / np.sum(valid_samples)
        else:
            mr_j = 0.0
        metrics.append(mr_j)
        print(f"  MR{j}: {mr_j:.4f} (valid samples: {np.sum(valid_samples)}, "
              f"correct: {np.sum(correct_samples) if np.sum(valid_samples) > 0 else 0})")
    return metrics


def absolute_false(Pre_Labels, test_target):
    num_instance, num_class = Pre_Labels.shape
    miss_pairs = np.sum(Pre_Labels != test_target)
    return miss_pairs / (num_class * num_instance)


def absolute_true(Pre_Labels, test_target):
    num_instance, _ = Pre_Labels.shape
    temp = sum(1 for i in range(num_instance) if np.array_equal(Pre_Labels[i], test_target[i]))
    return temp / num_instance


def accuracy(Pre_Labels, test_target):
    num_instance, num_class = Pre_Labels.shape
    temp = 0
    for i in range(num_instance):
        size_y = 0
        size_z = 0
        intersection = 0
        for j in range(num_class):
            if Pre_Labels[i][j] == 1:
                size_z += 1
            if test_target[i][j] == 1:
                size_y += 1
            if Pre_Labels[i][j] == 1 and test_target[i][j] == 1:
                intersection += 1
        if size_y != 0 and size_z != 0:
            temp += intersection / (size_y + size_z - intersection)
    return temp / num_instance


def aiming(Pre_Labels, test_target):
    num_instance, num_class = Pre_Labels.shape
    temp = 0
    for i in range(num_instance):
        size_z = 0
        intersection = 0
        for j in range(num_class):
            if Pre_Labels[i][j] == 1:
                size_z += 1
            if Pre_Labels[i][j] == 1 and test_target[i][j] == 1:
                intersection += 1
        if size_z != 0:
            temp += intersection / size_z
    return temp / num_instance


def coverage(Pre_Labels, test_target):
    num_instance, num_class = Pre_Labels.shape
    temp = 0
    for i in range(num_instance):
        size_y = 0
        intersection = 0
        for j in range(num_class):
            if test_target[i][j] == 1:
                size_y += 1
            if Pre_Labels[i][j] == 1 and test_target[i][j] == 1:
                intersection += 1
        if size_y != 0:
            temp += intersection / size_y
    return temp / num_instance

# Triple ensemble model class
class TripleEnsembleModel:
    """Ensemble model combining SVM, SAPP, and FNet"""

    def __init__(self, svm_weight=0.4, sapp_weight=0.3, fnet_weight=0.3, threshold=0.5):
        self.svm_weight = svm_weight
        self.sapp_weight = sapp_weight
        self.fnet_weight = fnet_weight
        self.threshold = threshold

    def set_weights(self, svm_weight, sapp_weight, fnet_weight):
        """Set ensemble weights"""
        total = svm_weight + sapp_weight + fnet_weight
        self.svm_weight = svm_weight / total
        self.sapp_weight = sapp_weight / total
        self.fnet_weight = fnet_weight / total

    def get_weights(self):
        """Get current weights"""
        return {
            'SVM': self.svm_weight,
            'SAPP': self.sapp_weight,
            'FNet': self.fnet_weight
        }

    def ensemble_predict(self, svm_preds, sapp_probs, fnet_probs, svm_scores=None):
        """
        Triple-model ensemble prediction.
        Parameters:
        - svm_preds: binary predictions from SVM {0, 1}
        - sapp_probs: probabilities from SAPP [0, 1]
        - fnet_probs: probabilities from FNet [0, 1]
        - svm_scores: decision_function scores from SVM (optional)
        Returns:
        - ensemble_probs: ensemble probabilities
        - ensemble_preds: ensemble binary predictions
        """
        if svm_scores is not None:
            # Convert SVM scores to probabilities (using sigmoid)
            svm_probs = 1 / (1 + np.exp(-svm_scores))
        else:
            # If no scores, use binary predictions directly as probabilities
            svm_probs = svm_preds.astype(np.float32)
        # Weighted average of three models
        ensemble_probs = (self.svm_weight * svm_probs +
                          self.sapp_weight * sapp_probs +
                          self.fnet_weight * fnet_probs)
        # Threshold to get final predictions
        ensemble_preds = (ensemble_probs > self.threshold).astype(np.int32)

        return ensemble_probs, ensemble_preds

def load_train_data(data_dir, num_classes, class_labels):
    """Load training data"""
    data_list = []
    label_list = []
    for i in range(num_classes):
        file_path = os.path.join(data_dir, f"class_{i}_data.npz")
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
        data = np.load(file_path)
        embeddings = data['embeddings']
        labels = np.array([class_labels[i]] * len(embeddings))
        data_list.append(embeddings)
        label_list.append(labels)
    if not data_list:
        raise ValueError(f"No valid data files found in {data_dir}.")
    X = np.concatenate(data_list, axis=0).astype(np.float32)
    y = np.concatenate(label_list, axis=0).astype(np.float32)
    return X, y


def load_test_data(data_dir, class_labels):
    """Load test data using exact filename pattern matching"""
    test_pattern = re.compile(r'^processed_\(\d+\)Test\d+\.npz$')
    test_files = sorted(
        [f for f in os.listdir(data_dir) if test_pattern.match(f)],
        key=lambda x: int(re.search(r'_\((\d+)\)', x).group(1)) - 1
    )

    print(f"  Found {len(test_files)} valid test files")
    data_list = []
    label_list = []
    for file in test_files:
        match = re.search(r'_\((\d+)\)', file)
        class_idx = int(match.group(1)) - 1
        data = np.load(os.path.join(data_dir, file))
        embeddings = data['embeddings']
        if embeddings.size == 0:
            print(f"  Warning: Skipping empty file - {file}")
            continue
        embeddings = embeddings.reshape(len(embeddings), -1)
        data_list.append(embeddings)
        label_list.extend([class_labels[class_idx]] * len(embeddings))

    X = np.concatenate(data_list).astype(np.float32)
    y = np.array(label_list).astype(np.float32)

    return X, y

# Main pipeline
def main():
    try:
        # Set random seed
        set_seed(RANDOM_SEED)
        print("=" * 70)
        print("SVM + SAPP + FNet Triple Ensemble - Independent Test Set Evaluation (Decision-Level Fusion)")
        print("=" * 70)
        print(f"\nUsing threshold: {THRESHOLD}")
        print("\nStep 1: Loading data...")
        # Load ESM2 training data
        print("  Loading ESM2 features...")
        X_train_esm2, y_train = load_train_data(TRAIN_DIR_FEATURE1, NUM_CLASSES, class_labels)
        print(f"  ESM2 training set shape: {X_train_esm2.shape}")
        # Load kmer training data
        print("  Loading kmer features...")
        X_train_kmer, _ = load_train_data(TRAIN_DIR_FEATURE2, NUM_CLASSES, class_labels)
        print(f"  kmer training set shape: {X_train_kmer.shape}")
        # Load test data
        print("\nStep 2: Loading test data...")
        print("  Loading ESM2 test features...")
        X_test_esm2, y_test = load_test_data(TEST_DIR_FEATURE1, class_labels)
        print(f"  ESM2 test set shape: {X_test_esm2.shape}")
        print("  Loading kmer test features...")
        X_test_kmer, _ = load_test_data(TEST_DIR_FEATURE2, class_labels)
        print(f"  kmer test set shape: {X_test_kmer.shape}")
        if X_test_esm2.shape[0] != X_test_kmer.shape[0]:
            raise ValueError(
                f"Test dataset sample counts are inconsistent! "
                f"ESM2: {X_test_esm2.shape[0]} samples, "
                f"kmer: {X_test_kmer.shape[0]} samples. "
                f"Check whether extra files (e.g. label_mapping.npz) need to be excluded."
            )
        print(f"  Validation passed: both test sets have {X_test_esm2.shape[0]} samples")
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # Create DataLoader generator
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED)
        # Train SVM model
        print("\nStep 3: Training SVM model...")
        print("  [1/2] Training SVM-ESM2 model...")
        svm_model_esm2 = create_optimized_svm(X_train_esm2, y_train.astype(np.int32))
        svm_model_esm2.fit(X_train_esm2, y_train.astype(np.int32))
        print("  [2/2] Training SVM-kmer model...")
        svm_model_kmer = create_optimized_svm(X_train_kmer, y_train.astype(np.int32))
        svm_model_kmer.fit(X_train_kmer, y_train.astype(np.int32))
        # SVM predictions
        svm_preds_esm2 = svm_model_esm2.predict(X_test_esm2)
        svm_preds_kmer = svm_model_kmer.predict(X_test_kmer)
        # Get SVM decision_function scores
        try:
            svm_scores_esm2 = np.zeros_like(svm_preds_esm2, dtype=np.float32)
            svm_scores_kmer = np.zeros_like(svm_preds_kmer, dtype=np.float32)
            for i, estimator in enumerate(svm_model_esm2.estimators_):
                svm_scores_esm2[:, i] = estimator.decision_function(X_test_esm2)
            for i, estimator in enumerate(svm_model_kmer.estimators_):
                svm_scores_kmer[:, i] = estimator.decision_function(X_test_kmer)
            # Average scores from two features
            svm_scores = (svm_scores_esm2 + svm_scores_kmer) / 2
        except:
            svm_scores = None
            print("  Warning: Could not get SVM decision_function scores; using binary predictions")
        # Average binary predictions
        svm_preds = ((svm_preds_esm2.astype(np.float32) + svm_preds_kmer.astype(np.float32)) / 2 > 0.5).astype(
            np.int32)
        # Train SAPP model
        print("\nStep 4: Training SAPP model...")
        # Convert to PyTorch tensors - ESM2
        print("  [1/2] Training SAPP-ESM2 model...")
        X_train_esm2_tensor = torch.FloatTensor(X_train_esm2).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_esm2_tensor = torch.FloatTensor(X_test_esm2).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        # Create data loaders - ESM2
        train_dataset_esm2 = TensorDataset(X_train_esm2_tensor, y_train_tensor)
        test_dataset_esm2 = TensorDataset(X_test_esm2_tensor, y_test_tensor)
        train_loader_esm2 = DataLoader(
            train_dataset_esm2, batch_size=SAPP_BATCH_SIZE_TRAIN, shuffle=True,
            worker_init_fn=seed_worker, generator=g
        )
        test_loader_esm2 = DataLoader(
            test_dataset_esm2, batch_size=SAPP_BATCH_SIZE_TEST, shuffle=False,
            worker_init_fn=seed_worker, generator=g
        )
        # Create SAPP model - ESM2
        sapp_model_esm2 = SAPPMultiLabelClassifier(
            input_dim=X_train_esm2.shape[1],
            device=device
        ).to(device)
        optimizer_esm2 = optim.AdamW(sapp_model_esm2.parameters(), lr=SAPP_LR, weight_decay=SAPP_WEIGHT_DECAY)
        criterion = nn.BCELoss()
        # Train SAPP-ESM2
        for epoch in range(SAPP_EPOCHS):
            sapp_model_esm2.train()
            for batch_x, batch_y in train_loader_esm2:
                optimizer_esm2.zero_grad()
                outputs = sapp_model_esm2(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer_esm2.step()
            if (epoch + 1) % 10 == 0:
                print(f"  SAPP-ESM2 Epoch {epoch + 1}/{SAPP_EPOCHS}")
        # Convert to PyTorch tensors - kmer
        print("  [2/2] Training SAPP-kmer model...")
        X_train_kmer_tensor = torch.FloatTensor(X_train_kmer).to(device)
        X_test_kmer_tensor = torch.FloatTensor(X_test_kmer).to(device)
        # Create data loaders - kmer
        train_dataset_kmer = TensorDataset(X_train_kmer_tensor, y_train_tensor)
        test_dataset_kmer = TensorDataset(X_test_kmer_tensor, y_test_tensor)
        train_loader_kmer = DataLoader(
            train_dataset_kmer, batch_size=SAPP_BATCH_SIZE_TRAIN, shuffle=True,
            worker_init_fn=seed_worker, generator=g
        )
        test_loader_kmer = DataLoader(
            test_dataset_kmer, batch_size=SAPP_BATCH_SIZE_TEST, shuffle=False,
            worker_init_fn=seed_worker, generator=g
        )
        # Create SAPP model - kmer
        sapp_model_kmer = SAPPMultiLabelClassifier(
            input_dim=X_train_kmer.shape[1],
            device=device
        ).to(device)
        optimizer_kmer = optim.AdamW(sapp_model_kmer.parameters(), lr=SAPP_LR, weight_decay=SAPP_WEIGHT_DECAY)
        # Train SAPP-kmer
        for epoch in range(SAPP_EPOCHS):
            sapp_model_kmer.train()
            for batch_x, batch_y in train_loader_kmer:
                optimizer_kmer.zero_grad()
                outputs = sapp_model_kmer(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer_kmer.step()
            if (epoch + 1) % 10 == 0:
                print(f"  SAPP-kmer Epoch {epoch + 1}/{SAPP_EPOCHS}")

        # SAPP predictions (probabilities)
        sapp_model_esm2.eval()
        sapp_model_kmer.eval()

        sapp_probs_esm2_list = []
        sapp_probs_kmer_list = []
        with torch.no_grad():
            for batch_x, _ in test_loader_esm2:
                outputs = sapp_model_esm2(batch_x)
                sapp_probs_esm2_list.append(outputs.cpu().numpy())
            for batch_x, _ in test_loader_kmer:
                outputs = sapp_model_kmer(batch_x)
                sapp_probs_kmer_list.append(outputs.cpu().numpy())
        sapp_probs_esm2 = np.concatenate(sapp_probs_esm2_list, axis=0)
        sapp_probs_kmer = np.concatenate(sapp_probs_kmer_list, axis=0)

        # Average probabilities from two features
        sapp_probs = (sapp_probs_esm2 + sapp_probs_kmer) / 2
        sapp_preds = (sapp_probs > THRESHOLD).astype(np.int32)

        # Train FNet model
        print("\nStep 5: Training FNet model...")

        print("  [1/2] Training FNet-ESM2 model...")
        fnet_model_esm2 = MultiOutputFNet(
            input_dim=X_train_esm2.shape[1],
            num_outputs=4,
            random_state=RANDOM_SEED
        )
        fnet_model_esm2.fit(X_train_esm2, y_train)

        print("  [2/2] Training FNet-kmer model...")
        fnet_model_kmer = MultiOutputFNet(
            input_dim=X_train_kmer.shape[1],
            num_outputs=4,
            random_state=RANDOM_SEED
        )
        fnet_model_kmer.fit(X_train_kmer, y_train)
        # FNet predictions (probabilities)
        fnet_probs_esm2 = fnet_model_esm2.predict_proba(X_test_esm2)
        fnet_probs_kmer = fnet_model_kmer.predict_proba(X_test_kmer)
        # Average probabilities from two features
        fnet_probs = (fnet_probs_esm2 + fnet_probs_kmer) / 2
        fnet_preds = (fnet_probs > THRESHOLD).astype(np.int32)
        # Triple-model ensemble prediction
        print("\nStep 6: Triple-model ensemble prediction...")
        y_test_np = y_test.astype(np.int32)
        # Use fixed weights
        best_weights = FIXED_WEIGHTS.copy()
        print(f"\n  Using fixed weights: SVM={best_weights['SVM']:.2f}, "
              f"SAPP={best_weights['SAPP']:.2f}, FNet={best_weights['FNet']:.2f}")
        # Ensemble prediction using weights
        ensemble = TripleEnsembleModel(
            best_weights['SVM'],
            best_weights['SAPP'],
            best_weights['FNet'],
            threshold=THRESHOLD
        )
        ensemble_probs, ensemble_preds = ensemble.ensemble_predict(
            svm_preds, sapp_probs, fnet_probs, svm_scores
        )

        # Compute metrics for each model
        print("\n" + "-" * 40)
        print("SVM model metrics:")
        print("-" * 40)
        mr_svm = calculate_custom_metrics(y_test_np, svm_preds)
        print(f"  Accuracy:       {accuracy(svm_preds,       y_test_np):.4f}")
        print(f"  Aiming:         {aiming(svm_preds,         y_test_np):.4f}")
        print(f"  Coverage:       {coverage(svm_preds,       y_test_np):.4f}")
        print(f"  Absolute_True:  {absolute_true(svm_preds,  y_test_np):.4f}")
        print(f"  Absolute_False: {absolute_false(svm_preds, y_test_np):.4f}")
        print("\n" + "-" * 40)
        print("SAPP model metrics:")
        print("-" * 40)
        mr_sapp = calculate_custom_metrics(y_test_np, sapp_preds)
        print(f"  Accuracy:       {accuracy(sapp_preds,       y_test_np):.4f}")
        print(f"  Aiming:         {aiming(sapp_preds,         y_test_np):.4f}")
        print(f"  Coverage:       {coverage(sapp_preds,       y_test_np):.4f}")
        print(f"  Absolute_True:  {absolute_true(sapp_preds,  y_test_np):.4f}")
        print(f"  Absolute_False: {absolute_false(sapp_preds, y_test_np):.4f}")
        print("\n" + "-" * 40)
        print("FNet model metrics:")
        print("-" * 40)
        mr_fnet = calculate_custom_metrics(y_test_np, fnet_preds)
        print(f"  Accuracy:       {accuracy(fnet_preds,       y_test_np):.4f}")
        print(f"  Aiming:         {aiming(fnet_preds,         y_test_np):.4f}")
        print(f"  Coverage:       {coverage(fnet_preds,       y_test_np):.4f}")
        print(f"  Absolute_True:  {absolute_true(fnet_preds,  y_test_np):.4f}")
        print(f"  Absolute_False: {absolute_false(fnet_preds, y_test_np):.4f}")
        print("\n" + "-" * 40)
        print("Triple-model ensemble metrics:")
        print("-" * 40)
        mr_ensemble = calculate_custom_metrics(y_test_np, ensemble_preds)
        print(f"  Accuracy:       {accuracy(ensemble_preds,       y_test_np):.4f}")
        print(f"  Aiming:         {aiming(ensemble_preds,         y_test_np):.4f}")
        print(f"  Coverage:       {coverage(ensemble_preds,       y_test_np):.4f}")
        print(f"  Absolute_True:  {absolute_true(ensemble_preds,  y_test_np):.4f}")
        print(f"  Absolute_False: {absolute_false(ensemble_preds, y_test_np):.4f}")
        print("\n" + "=" * 70)
        print("Independent Test Set Evaluation Summary")
        print("=" * 70)
        print("\n" + "=" * 80)
        print("Model performance comparison table:")
        print("=" * 80)
        print(f"{'Metric':<18}{'SVM':<12}{'SAPP':<12}{'FNet':<12}{'Ensemble':<12}{'Best Model':<12}")
        print("-" * 80)

        metrics_data = {
            "SVM":      {"MR": mr_svm,      "AF": absolute_false(svm_preds,      y_test_np), "AT": absolute_true(svm_preds,      y_test_np), "Acc": accuracy(svm_preds,      y_test_np), "Aim": aiming(svm_preds,      y_test_np), "Cov": coverage(svm_preds,      y_test_np)},
            "SAPP":     {"MR": mr_sapp,     "AF": absolute_false(sapp_preds,     y_test_np), "AT": absolute_true(sapp_preds,     y_test_np), "Acc": accuracy(sapp_preds,     y_test_np), "Aim": aiming(sapp_preds,     y_test_np), "Cov": coverage(sapp_preds,     y_test_np)},
            "FNet":     {"MR": mr_fnet,     "AF": absolute_false(fnet_preds,     y_test_np), "AT": absolute_true(fnet_preds,     y_test_np), "Acc": accuracy(fnet_preds,     y_test_np), "Aim": aiming(fnet_preds,     y_test_np), "Cov": coverage(fnet_preds,     y_test_np)},
            "Ensemble": {"MR": mr_ensemble, "AF": absolute_false(ensemble_preds, y_test_np), "AT": absolute_true(ensemble_preds, y_test_np), "Acc": accuracy(ensemble_preds, y_test_np), "Aim": aiming(ensemble_preds, y_test_np), "Cov": coverage(ensemble_preds, y_test_np)},
        }
        for metric_key, label in [("Acc","Accuracy"), ("Aim","Aiming"), ("Cov","Coverage"), ("AT","Absolute_True")]:
            vals = {m: metrics_data[m][metric_key] for m in ["SVM","SAPP","FNet","Ensemble"]}
            best = max(vals, key=vals.get)
            print(f"{label:<18}{vals['SVM']:<12.4f}{vals['SAPP']:<12.4f}{vals['FNet']:<12.4f}{vals['Ensemble']:<12.4f}{best:<12}")
        for i in range(4):
            metric_name = f"MR{i+1}"
            vals = {m: metrics_data[m]["MR"][i] for m in ["SVM","SAPP","FNet","Ensemble"]}
            best = max(vals, key=vals.get)
            print(f"{metric_name:<18}{vals['SVM']:<12.4f}{vals['SAPP']:<12.4f}{vals['FNet']:<12.4f}{vals['Ensemble']:<12.4f}{best:<12}")

        vals = {m: metrics_data[m]["AF"] for m in ["SVM","SAPP","FNet","Ensemble"]}
        best = min(vals, key=vals.get)
        print(f"{'Absolute_False':<18}{vals['SVM']:<12.4f}{vals['SAPP']:<12.4f}{vals['FNet']:<12.4f}{vals['Ensemble']:<12.4f}{best:<12}")
        print("-" * 80)

    except Exception as e:
        import traceback
        print(f"\nError in main process: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
