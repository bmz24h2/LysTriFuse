import numpy as np
import os
import re
import json
from glob import glob
from collections import Counter
from sklearn.neighbors import NearestNeighbors


class ENNDataLoader:
    def __init__(self, dataset_dir, class_labels, min_samples=1180, k=3):
        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.transformer = LabelTransformer(class_labels)
        self.min_samples = min_samples  # Target minimum sample count
        self.k = k  # ENN algorithm parameter

    def load_and_process(self):
        # Load raw data
        data, labels = self._load_raw_data()
        y_encoded = self.transformer.encode(labels)

        # Perform ENN downsampling
        data_resampled, y_resampled = self._enn_downsample(data, y_encoded)
        # Convert back to original label format
        final_labels = self.transformer.decode(y_resampled)
        return data_resampled, final_labels

    def _enn_downsample(self, X, y):
        """Core logic for ENN downsampling"""
        # Get sample count per class
        class_counts = Counter(y)
        # Build sampling strategy (only for classes exceeding the threshold)
        sampling_strategy = {
            cls: self.min_samples
            for cls, count in class_counts.items()
            if count > self.min_samples
        }
        # If no classes need downsampling, return original data
        if not sampling_strategy:
            return X, y
        # Initialize nearest-neighbor model
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(X)
        # Get indices of classes to process
        process_classes = list(sampling_strategy.keys())
        mask = np.isin(y, process_classes)
        # Run ENN algorithm
        keep_indices = []
        for i in range(X.shape[0]):
            if mask[i]:  # Process target-class samples only
                distances, indices = nn.kneighbors([X[i]])
                nearest_labels = y[indices[0]]
                current_label = y[i]
                # Keep samples consistent with the majority class
                if np.sum(nearest_labels == current_label) > self.k // 2:
                    keep_indices.append(i)
            else:  # Retain all samples from non-target classes
                keep_indices.append(i)

        return X[keep_indices], y[keep_indices]

    def _load_raw_data(self):
        """Improved data loading method"""
        all_data = []
        all_labels = []
        files = sorted(glob(os.path.join(self.dataset_dir, 'class_*_data.npz')),
                       key=lambda x: int(re.search(r'class_(\d+)_data\.npz', x).group(1)))
        if len(files) != len(self.class_labels):
            raise ValueError(f"Found {len(files)} data files, but {len(self.class_labels)} classes are required")
        for file_path in files:
            class_num = int(re.search(r'class_(\d+)_data\.npz', file_path).group(1))
            with np.load(file_path) as npz_data:
                class_data = npz_data['embeddings']
                all_data.append(class_data)
                all_labels.extend([self.class_labels[class_num]] * class_data.shape[0])
        return np.vstack(all_data), np.array(all_labels)


class StructuredDataSaver:
    def __init__(self, output_dir="ESM2_OneHot_KPCA1_ENN1_dataset_100d_20d_5"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.class_counts = {}  # Sample counter

    def save_by_class(self, data, labels, class_labels):
        y_encoded = LabelTransformer(class_labels).encode(labels)
        for class_id in range(len(class_labels)):
            class_indices = np.where(y_encoded == class_id)[0]
            class_data = data[class_indices]
            self.class_counts[class_id] = len(class_data)  # Record sample count
            names = [f"class_{class_id}_sample_{i:04d}" for i in range(len(class_data))]
            filename = f"class_{class_id}_data.npz"
            filepath = os.path.join(self.output_dir, filename)
            np.savez(filepath, names=np.array(names), embeddings=class_data)
            print(f"Class {class_id}: saved {len(class_data)} samples to: {filepath}")
        # Save label mapping
        label_mapping_filepath = os.path.join(self.output_dir, "label_mapping.npz")
        np.savez(label_mapping_filepath,
                 class_labels=np.array(class_labels, dtype=np.int8),
                 label_mapping=json.dumps({i: list(l) for i, l in enumerate(class_labels)}))
        # Save statistics
        self._save_statistics()

    def _save_statistics(self):
        """Save statistics to file"""
        txt_path = os.path.join(self.output_dir, 'class_counts.txt')
        json_path = os.path.join(self.output_dir, 'class_counts.json')
        # Text format
        with open(txt_path, 'w') as f:
            for cls, cnt in sorted(self.class_counts.items()):
                f.write(f"Class {cls}: {cnt} samples\n")
        # JSON format
        with open(json_path, 'w') as f:
            json.dump(self.class_counts, f, indent=4)
        print(f"\nSample statistics files generated:")
        print(f"├── {txt_path}")
        print(f"└── {json_path}")


class LabelTransformer:
    """Improved label converter (supports multi-label encoding)"""

    def __init__(self, class_labels):
        self.class_labels = [tuple(label) for label in class_labels]
        self.label_to_int = {label: idx for idx, label in enumerate(self.class_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}

    def encode(self, labels):
        return np.array([self.label_to_int[tuple(label)] for label in labels])

    def decode(self, encoded_labels):
        return np.array([list(self.int_to_label[label]) for label in encoded_labels])


if __name__ == "__main__":
    CONFIG = {
        'dataset_dir': './ESM2_OneHot_KPCA1_ENN1_dataset_100d_20d_4',
        'class_labels': [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
            [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]
        ],
        'min_samples': 1180,
        'k': 3
    }

    try:
        # Initialize loader
        loader = ENNDataLoader(
            dataset_dir=CONFIG['dataset_dir'],
            class_labels=CONFIG['class_labels'],
            min_samples=CONFIG['min_samples'],
            k=CONFIG['k']
        )
        # Execute processing pipeline
        final_data, final_labels = loader.load_and_process()
        # Verify results
        unique, counts = np.unique(loader.transformer.encode(final_labels), return_counts=True)
        print("\nFinal class distribution:")
        for cls, cnt in zip(unique, counts):
            status = "√" if cnt >= CONFIG['min_samples'] else f"x (current: {cnt})"
            print(f"Class {cls}: {status}")
        saver = StructuredDataSaver()
        saver.save_by_class(final_data, final_labels, CONFIG['class_labels'])
        # Validate statistics
        print("\nStatistics validation:")
        stats_file = os.path.join("ESM2_OneHot_100d_20d/ESM2_OneHot_KPCA1_ENN1_dataset_100d_20d_5", "class_counts.txt")
        with open(stats_file, 'r') as f:
            print(f.read())

    except Exception as e:
        print(f"Processing failed: {str(e)}")
