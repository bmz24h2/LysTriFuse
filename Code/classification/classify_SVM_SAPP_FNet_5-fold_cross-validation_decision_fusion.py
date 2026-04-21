import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, TensorDataset
import math

FIXED_WEIGHTS = {
    'SVM': 0.6,
    'SAPP': 0.1,
    'FNet': 0.3
}

THRESHOLD = 0.5

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

num_classes = 11

# Separate feature data paths
input_dir_feature1 = "Test_22_KPCA_CC_OSS_ENN_OVER_UNDER_balanced_dataset_22"
input_dir_feature2 = "DACC_data1/DACC_KPCA1_ENN3_OSS_CC_Train35_dataset"

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

class FNetLayer(nn.Module):
    """FNet Layer with Fourier Transform"""
    def __init__(self, hidden_dim, dropout=0.1):
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

    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.1):
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
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, dropout=0.1,
                 lr=0.001, epochs=50, batch_size=64, random_state=None):
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
    def __init__(self, input_dim, num_outputs=4, hidden_dim=512, num_layers=2,
                 dropout=0.1, lr=0.001, epochs=50, batch_size=64, random_state=None):
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
    def __init__(self, input_dim, num_labels=4, hidden_dim=256, n_layers=2,
                 attn_heads=4, feed_forward_dim=758, dropout=0.2, device='cuda'):
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

def create_optimized_svm(X_train, y_train):
    """Create and optimize SVM classifier"""
    base_svm = LinearSVC(
        dual=False,
        random_state=42,
        max_iter=10000,
        tol=1e-4
    )
    ovr_svm = OneVsRestClassifier(base_svm)
    param_grid = [
        {
            'estimator__C': [0.1, 1, 10],
            'estimator__penalty': ['l1'],
            'estimator__loss': ['squared_hinge']
        },
        {
            'estimator__C': [0.1, 1, 10],
            'estimator__penalty': ['l2'],
            'estimator__loss': ['squared_hinge']
        }
    ]
    grid_search = GridSearchCV(
        estimator=ovr_svm,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"  SVM best params: {grid_search.best_params_}")
    print(f"  SVM best score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

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

def load_data(data_dir, num_classes, class_labels):
    """Load data"""
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


# ===============================
# Main pipeline
# ===============================
def main():
    try:
        # Set random seed
        set_seed(42)
        print("=" * 70)
        print("SVM + SAPP + FNet Triple Ensemble - 5-Fold Cross Validation (Decision-Level Fusion)")
        print("=" * 70)
        print(f"\nUsing threshold: {THRESHOLD}")
        print("\nStep 1: Loading data...")
        # Load ESM2 data
        print("  Loading ESM2 features...")
        X_esm2, y = load_data(input_dir_feature1, num_classes, class_labels)
        print(f"  ESM2 data shape: {X_esm2.shape}")
        # Load kmer data
        print("  Loading kmer features...")
        X_kmer, _ = load_data(input_dir_feature2, num_classes, class_labels)
        print(f"  kmer data shape: {X_kmer.shape}")
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # 5-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # Store results for each model and ensemble
        metrics_results = {
            "SVM": {"MR": [], "Absolute_False": [], "Absolute_True": [],
                    "Accuracy": [], "Aiming": [], "Coverage": []},
            "SAPP": {"MR": [], "Absolute_False": [], "Absolute_True": [],
                     "Accuracy": [], "Aiming": [], "Coverage": []},
            "FNet": {"MR": [], "Absolute_False": [], "Absolute_True": [],
                     "Accuracy": [], "Aiming": [], "Coverage": []},
            "Ensemble": {"MR": [], "Absolute_False": [], "Absolute_True": [],
                         "Accuracy": [], "Aiming": [], "Coverage": []}
        }

        # Store weights per fold
        fold_weights = []
        # Create DataLoader generator
        g = torch.Generator()
        g.manual_seed(42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_esm2)):
            print(f"\n{'=' * 70}")
            print(f"Fold {fold + 1}")
            print('=' * 70)
            # Data split
            X_train_esm2, X_test_esm2 = X_esm2[train_idx], X_esm2[test_idx]
            X_train_kmer, X_test_kmer = X_kmer[train_idx], X_kmer[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # ===============================
            # Train SVM model
            # ===============================
            print("\n[1/4] Training SVM model...")
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
            print("\n[2/4] Training SAPP model...")
            print("  [1/2] Training SAPP-ESM2 model...")
            X_train_esm2_tensor = torch.FloatTensor(X_train_esm2).to(device)
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            X_test_esm2_tensor = torch.FloatTensor(X_test_esm2).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)

            train_dataset_esm2 = TensorDataset(X_train_esm2_tensor, y_train_tensor)
            test_dataset_esm2 = TensorDataset(X_test_esm2_tensor, y_test_tensor)
            train_loader_esm2 = DataLoader(
                train_dataset_esm2, batch_size=64, shuffle=True,
                worker_init_fn=seed_worker, generator=g
            )
            test_loader_esm2 = DataLoader(
                test_dataset_esm2, batch_size=128, shuffle=False,
                worker_init_fn=seed_worker, generator=g
            )

            sapp_model_esm2 = SAPPMultiLabelClassifier(
                input_dim=X_train_esm2.shape[1],
                num_labels=4,
                hidden_dim=256,
                n_layers=2,
                attn_heads=4,
                feed_forward_dim=758,
                dropout=0.2,
                device=device
            ).to(device)

            optimizer_esm2 = optim.AdamW(sapp_model_esm2.parameters(), lr=0.0005, weight_decay=1e-6)
            criterion = nn.BCELoss()

            # Train SAPP-ESM2
            num_epochs = 50
            for epoch in range(num_epochs):
                sapp_model_esm2.train()
                for batch_x, batch_y in train_loader_esm2:
                    optimizer_esm2.zero_grad()
                    outputs = sapp_model_esm2(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer_esm2.step()
                if (epoch + 1) % 10 == 0:
                    print(f"  SAPP-ESM2 Epoch {epoch + 1}/{num_epochs}")
            # Convert to PyTorch tensors - kmer
            print("  [2/2] Training SAPP-kmer model...")
            X_train_kmer_tensor = torch.FloatTensor(X_train_kmer).to(device)
            X_test_kmer_tensor = torch.FloatTensor(X_test_kmer).to(device)
            # Create data loaders - kmer
            train_dataset_kmer = TensorDataset(X_train_kmer_tensor, y_train_tensor)
            test_dataset_kmer = TensorDataset(X_test_kmer_tensor, y_test_tensor)
            train_loader_kmer = DataLoader(
                train_dataset_kmer, batch_size=64, shuffle=True,
                worker_init_fn=seed_worker, generator=g
            )
            test_loader_kmer = DataLoader(
                test_dataset_kmer, batch_size=128, shuffle=False,
                worker_init_fn=seed_worker, generator=g
            )
            # Create SAPP model - kmer
            sapp_model_kmer = SAPPMultiLabelClassifier(
                input_dim=X_train_kmer.shape[1],
                num_labels=4,
                hidden_dim=256,
                n_layers=2,
                attn_heads=4,
                feed_forward_dim=758,
                dropout=0.2,
                device=device
            ).to(device)
            optimizer_kmer = optim.AdamW(sapp_model_kmer.parameters(), lr=0.0005, weight_decay=1e-6)
            # Train SAPP-kmer
            for epoch in range(num_epochs):
                sapp_model_kmer.train()
                for batch_x, batch_y in train_loader_kmer:
                    optimizer_kmer.zero_grad()
                    outputs = sapp_model_kmer(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer_kmer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"  SAPP-kmer Epoch {epoch + 1}/{num_epochs}")

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


            print("\n[3/4] Training FNet model...")

            print("  [1/2] Training FNet-ESM2 model...")
            fnet_model_esm2 = MultiOutputFNet(
                input_dim=X_train_esm2.shape[1],
                num_outputs=4,
                hidden_dim=512,
                num_layers=2,
                dropout=0.1,
                lr=0.001,
                epochs=50,
                batch_size=64,
                random_state=42
            )
            fnet_model_esm2.fit(X_train_esm2, y_train)

            print("  [2/2] Training FNet-kmer model...")
            fnet_model_kmer = MultiOutputFNet(
                input_dim=X_train_kmer.shape[1],
                num_outputs=4,
                hidden_dim=512,
                num_layers=2,
                dropout=0.1,
                lr=0.001,
                epochs=50,
                batch_size=64,
                random_state=42
            )
            fnet_model_kmer.fit(X_train_kmer, y_train)

            # FNet predictions (probabilities)
            fnet_probs_esm2 = fnet_model_esm2.predict_proba(X_test_esm2)
            fnet_probs_kmer = fnet_model_kmer.predict_proba(X_test_kmer)

            # Average probabilities from two features
            fnet_probs = (fnet_probs_esm2 + fnet_probs_kmer) / 2
            fnet_preds = (fnet_probs > THRESHOLD).astype(np.int32)

            # ===============================
            # Triple-model ensemble prediction
            # ===============================
            print("\n[4/4] Triple-model ensemble prediction...")

            y_test_np = y_test.astype(np.int32)

            # Use fixed weights
            best_weights = FIXED_WEIGHTS.copy()
            fold_weights.append(best_weights)

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


            print("\n" + "-" * 40)
            print("SVM model metrics:")
            print("-" * 40)
            mr_svm = calculate_custom_metrics(y_test_np, svm_preds)
            metrics_results["SVM"]["MR"].append(mr_svm)
            metrics_results["SVM"]["Absolute_False"].append(absolute_false(svm_preds, y_test_np))
            metrics_results["SVM"]["Absolute_True"].append(absolute_true(svm_preds, y_test_np))
            metrics_results["SVM"]["Accuracy"].append(accuracy(svm_preds, y_test_np))
            metrics_results["SVM"]["Aiming"].append(aiming(svm_preds, y_test_np))
            metrics_results["SVM"]["Coverage"].append(coverage(svm_preds, y_test_np))

            print("\n" + "-" * 40)
            print("SAPP model metrics:")
            print("-" * 40)
            mr_sapp = calculate_custom_metrics(y_test_np, sapp_preds)
            metrics_results["SAPP"]["MR"].append(mr_sapp)
            metrics_results["SAPP"]["Absolute_False"].append(absolute_false(sapp_preds, y_test_np))
            metrics_results["SAPP"]["Absolute_True"].append(absolute_true(sapp_preds, y_test_np))
            metrics_results["SAPP"]["Accuracy"].append(accuracy(sapp_preds, y_test_np))
            metrics_results["SAPP"]["Aiming"].append(aiming(sapp_preds, y_test_np))
            metrics_results["SAPP"]["Coverage"].append(coverage(sapp_preds, y_test_np))

            print("\n" + "-" * 40)
            print("FNet model metrics:")
            print("-" * 40)
            mr_fnet = calculate_custom_metrics(y_test_np, fnet_preds)
            metrics_results["FNet"]["MR"].append(mr_fnet)
            metrics_results["FNet"]["Absolute_False"].append(absolute_false(fnet_preds, y_test_np))
            metrics_results["FNet"]["Absolute_True"].append(absolute_true(fnet_preds, y_test_np))
            metrics_results["FNet"]["Accuracy"].append(accuracy(fnet_preds, y_test_np))
            metrics_results["FNet"]["Aiming"].append(aiming(fnet_preds, y_test_np))
            metrics_results["FNet"]["Coverage"].append(coverage(fnet_preds, y_test_np))

            print("\n" + "-" * 40)
            print("Triple-model ensemble metrics:")
            print("-" * 40)
            mr_ensemble = calculate_custom_metrics(y_test_np, ensemble_preds)
            metrics_results["Ensemble"]["MR"].append(mr_ensemble)
            metrics_results["Ensemble"]["Absolute_False"].append(absolute_false(ensemble_preds, y_test_np))
            metrics_results["Ensemble"]["Absolute_True"].append(absolute_true(ensemble_preds, y_test_np))
            metrics_results["Ensemble"]["Accuracy"].append(accuracy(ensemble_preds, y_test_np))
            metrics_results["Ensemble"]["Aiming"].append(aiming(ensemble_preds, y_test_np))
            metrics_results["Ensemble"]["Coverage"].append(coverage(ensemble_preds, y_test_np))

        # Summarize results
        print("\n" + "=" * 70)
        print("5-Fold Cross Validation Summary")
        print("=" * 70)

        # Compute average weights
        avg_svm_weight = np.mean([w['SVM'] for w in fold_weights])
        avg_sapp_weight = np.mean([w['SAPP'] for w in fold_weights])
        avg_fnet_weight = np.mean([w['FNet'] for w in fold_weights])

        print("\n" + "-" * 50)
        print("Ensemble weight statistics:")
        print("-" * 50)
        print(f"{'Fold':<10}{'SVM Weight':<12}{'SAPP Weight':<12}{'FNet Weight':<12}")
        print("-" * 50)
        for i, w in enumerate(fold_weights):
            print(f"Fold {i + 1:<5}{w['SVM']:<12.2f}{w['SAPP']:<12.2f}{w['FNet']:<12.2f}")
        print("-" * 50)
        print(f"{'Average':<10}{avg_svm_weight:<12.2f}{avg_sapp_weight:<12.2f}{avg_fnet_weight:<12.2f}")

        # Print average metrics for each model
        for model_name in ["SVM", "SAPP", "FNet", "Ensemble"]:
            print(f"\n{'=' * 40}")
            print(f"{model_name} average metrics:")
            print("=" * 40)

            mr_matrix = np.array(metrics_results[model_name]["MR"])
            mean_mr = np.mean(mr_matrix, axis=0)

            mean_metrics = {
                **{f"MR{i + 1}": mean_mr[i] for i in range(4)},
                "Absolute_False": np.mean(metrics_results[model_name]["Absolute_False"]),
                "Absolute_True": np.mean(metrics_results[model_name]["Absolute_True"]),
                "Accuracy": np.mean(metrics_results[model_name]["Accuracy"]),
                "Aiming": np.mean(metrics_results[model_name]["Aiming"]),
                "Coverage": np.mean(metrics_results[model_name]["Coverage"])
            }

            for key, value in mean_metrics.items():
                print(f"{key}: {value:.4f}")

        # Print comparison table
        print("\n" + "=" * 80)
        print("Model performance comparison table:")
        print("=" * 80)
        print(f"{'Metric':<18}{'SVM':<12}{'SAPP':<12}{'FNet':<12}{'Ensemble':<12}{'Best Model':<12}")
        print("-" * 80)

        metric_names = ["Accuracy", "Aiming", "Coverage", "Absolute_True"]
        for metric in metric_names:
            svm_val = np.mean(metrics_results["SVM"][metric])
            sapp_val = np.mean(metrics_results["SAPP"][metric])
            fnet_val = np.mean(metrics_results["FNet"][metric])
            ens_val = np.mean(metrics_results["Ensemble"][metric])

            values = {"SVM": svm_val, "SAPP": sapp_val, "FNet": fnet_val, "Ensemble": ens_val}
            best_model = max(values, key=values.get)

            print(f"{metric:<18}{svm_val:<12.4f}{sapp_val:<12.4f}{fnet_val:<12.4f}{ens_val:<12.4f}{best_model:<12}")

        # MR metric comparison
        for i in range(4):
            metric_name = f"MR{i + 1}"
            svm_val = np.mean(np.array(metrics_results["SVM"]["MR"])[:, i])
            sapp_val = np.mean(np.array(metrics_results["SAPP"]["MR"])[:, i])
            fnet_val = np.mean(np.array(metrics_results["FNet"]["MR"])[:, i])
            ens_val = np.mean(np.array(metrics_results["Ensemble"]["MR"])[:, i])

            values = {"SVM": svm_val, "SAPP": sapp_val, "FNet": fnet_val, "Ensemble": ens_val}
            best_model = max(values, key=values.get)

            print(
                f"{metric_name:<18}{svm_val:<12.4f}{sapp_val:<12.4f}{fnet_val:<12.4f}{ens_val:<12.4f}{best_model:<12}")

        # Absolute_False (lower is better)
        print("-" * 80)
        metric = "Absolute_False"
        svm_val = np.mean(metrics_results["SVM"][metric])
        sapp_val = np.mean(metrics_results["SAPP"][metric])
        fnet_val = np.mean(metrics_results["FNet"][metric])
        ens_val = np.mean(metrics_results["Ensemble"][metric])
        # Negate because lower is better
        values = {"SVM": -svm_val, "SAPP": -sapp_val, "FNet": -fnet_val, "Ensemble": -ens_val}
        best_model = max(values, key=values.get)
        print(f"{metric:<18}{svm_val:<12.4f}{sapp_val:<12.4f}{fnet_val:<12.4f}{ens_val:<12.4f}{best_model:<12}")

    except Exception as e:
        import traceback
        print(f"\nError in main process: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
