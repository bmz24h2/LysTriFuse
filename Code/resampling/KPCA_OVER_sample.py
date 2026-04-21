import json
import numpy as np
import os
import re
from glob import glob
from collections import Counter

# KPCA algorithm
def kpca_oversample(data, labels, sampling_strategy, n_components=3, kernel='rbf', gamma=None, random_state=None):
    from sklearn.decomposition import KernelPCA
    from sklearn.neighbors import NearestNeighbors
    labels = np.array(labels, dtype=int)
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma,
                     fit_inverse_transform=True, random_state=random_state)
    resampled_data = []
    resampled_labels = []
    for label, target_count in sampling_strategy.items():
        indices = np.where(labels == label)[0]
        data_label = data[indices]
        if len(indices) == 0:
            continue
        # KPCA processing
        kpca.fit(data_label)
        synthetic_samples = kpca.inverse_transform(kpca.transform(data_label))
        # Generate interpolated samples
        num_synthetic = max(0, target_count - len(indices))
        if num_synthetic > 0:
            nn = NearestNeighbors(n_neighbors=min(len(indices), 5))
            nn.fit(data_label)
            neighbors = nn.kneighbors(data_label, return_distance=False)
            synthetic = []
            for _ in range(num_synthetic):
                idx1, idx2 = np.random.choice(len(data_label), 2)
                weight = np.random.rand()
                new_sample = data_label[idx1] * weight + data_label[idx2] * (1 - weight)
                synthetic.append(new_sample)
            if len(synthetic) > 0:
                synthetic_samples = np.vstack([synthetic_samples, synthetic])
        # Ensure sample count is accurate
        final_samples = np.vstack([data_label, synthetic_samples])[:target_count]
        resampled_data.append(final_samples)
        resampled_labels.extend([label] * target_count)
    return np.vstack(resampled_data), np.array(resampled_labels)

# Modified data loader
class DataLoader:
    def __init__(self, dataset_dir, class_labels):
        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.transformer = LabelTransformer(class_labels)
        self.min_samples = 1180  # Target sample count

    def load_and_process(self):
        # Load raw data
        data, labels = self._load_raw_data()
        y_encoded = self.transformer.encode(labels)
        # Build sampling strategy
        class_counts = dict(Counter(y_encoded))
        sampling_strategy = {
            cls: max(count, self.min_samples)
            for cls, count in class_counts.items()
        }
        # Oversample using KPCA
        resampled_data, resampled_labels = kpca_oversample(
            data, y_encoded, sampling_strategy,
            n_components=3, kernel='rbf', gamma=0.1, random_state=42
        )
        return resampled_data, self.transformer.decode(resampled_labels)

    def _load_raw_data(self):
        all_data = []
        all_labels = []
        # Improved file matching pattern
        files = sorted(glob(os.path.join(self.dataset_dir, 'processed_(*)Train*.npz')),
                       key=lambda x: int(re.search(r'processed_\((\d+)\)', x).group(1)))
        if len(files) != len(self.class_labels):
            raise ValueError(f"Found {len(files)} data files, but {len(self.class_labels)} classes are required")
        for idx, file_path in enumerate(files):
            with np.load(file_path) as npz_data:
                class_data = npz_data['embeddings']  # Read embeddings field
                all_data.append(class_data)
                all_labels.extend([self.class_labels[idx]] * class_data.shape[0])
        return np.vstack(all_data), np.array(all_labels)

# Enhanced file saver
class StructuredDataSaver:
    def __init__(self, output_dir="ESM2_OneHot_KPCA1_dataset_100d_20d"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_by_class(self, data, labels, class_labels):
        y_encoded = LabelTransformer(class_labels).encode(labels)
        for class_id in range(len(class_labels)):
            class_indices = np.where(y_encoded == class_id)[0]
            class_data = data[class_indices]
            # Generate sample names
            names = [f"class_{class_id}_sample_{i:04d}" for i in range(len(class_data))]
            # Save as npz file containing names and embeddings
            filename = f"class_{class_id}_data.npz"
            filepath = os.path.join(self.output_dir, filename)
            np.savez(filepath,
                     names=np.array(names),
                     embeddings=class_data)
            print(f"Saved class {class_id}: {len(class_data)} samples | filename: {filename}")

        label_mapping_filepath = os.path.join(self.output_dir, "label_mapping.npz")
        # Save label mapping
        label_mapping = {i: list(label) for i, label in enumerate(class_labels)}
        # Convert to NumPy-compatible types
        np.savez(label_mapping_filepath,
                 class_labels=np.array(class_labels, dtype=np.int8),  # Use numeric type
                 label_mapping=json.dumps(label_mapping))  # Convert dict to JSON string

# Label transformer
class LabelTransformer:
    def __init__(self, class_labels):
        self.class_labels = [tuple(label) for label in class_labels]
        self.label_to_int = {label: idx for idx, label in enumerate(self.class_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}

    def encode(self, labels):
        return np.array([self.label_to_int[tuple(label)] for label in labels])

    def decode(self, encoded_labels):
        return np.array([list(self.int_to_label[label]) for label in encoded_labels])


# ================== Usage example ==================
if __name__ == "__main__":
    CONFIG = {
        'dataset_dir': './ESM2_OneHot_FE_Train_dataset_100d_20d',
        'class_labels': [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
            [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]
        ]
    }
    # Initialize loader
    loader = DataLoader(CONFIG['dataset_dir'], CONFIG['class_labels'])
    try:
        # Execute processing pipeline
        final_data, final_labels = loader.load_and_process()
        # Summarize results
        unique, counts = np.unique(loader.transformer.encode(final_labels), return_counts=True)
        print("\nFinal class distribution:")
        for cls, cnt in zip(unique, counts):
            status = "√" if cnt >= 1180 else "x"
            print(f"Class {cls}: {cnt} samples {status}")
        # Save results
        saver = StructuredDataSaver()
        saver.save_by_class(final_data, final_labels, CONFIG['class_labels'])
        # Print file structure
        print("\nGenerated file structure:")
        print(f"/ESM2_OneHot_KPCA1_dataset_100d_20d")
        for i in range(11):
            print(f"├── class_{i}_data.npz")
        print(f"└── label_mapping.npz")

    except Exception as e:
        print(f"Processing failed: {str(e)}")
