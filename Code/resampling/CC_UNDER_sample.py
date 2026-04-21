import json
import numpy as np
import os
import re
from glob import glob
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter

def cluster_centroids_downsample(data, labels, sampling_strategy, n_init=10, voting='auto'):
    """
    Downsample using ClusterCentroids.
    :param data: feature data (n_samples, n_features)
    :param labels: integer-encoded labels (n_samples,)
    :param sampling_strategy: target sample count per class, e.g. {0: 100, 1: 200}
    :return: downsampled data and labels
    """
    under = ClusterCentroids(
        sampling_strategy=sampling_strategy,
        random_state=42,
        voting=voting,
        estimator=MiniBatchKMeans(n_init=n_init, random_state=42, batch_size=2048)
    )
    x_resampled, y_resampled = under.fit_resample(data, labels)
    return x_resampled, y_resampled


class DataLoader:
    def __init__(self, dataset_dir, class_labels, min_samples=1180, n_init=10, voting='auto'):
        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.transformer = LabelTransformer(class_labels)
        self.min_samples = min_samples
        self.n_init = n_init  # ClusterCentroids parameter
        self.voting = voting  # ClusterCentroids parameter

    def load_and_process(self):
        # Load raw data
        data, labels = self._load_raw_data()
        y_encoded = self.transformer.encode(labels)

        # Get class distribution
        unique_classes = np.unique(y_encoded)
        class_counts = Counter(y_encoded)

        # ClusterCentroids downsampling
        sampling_strategy = {}
        for cls in unique_classes:
            if class_counts[cls] > self.min_samples:
                sampling_strategy[cls] = self.min_samples

        if sampling_strategy:
            data, y_encoded = cluster_centroids_downsample(
                data=data,
                labels=y_encoded,
                sampling_strategy=sampling_strategy,
                n_init=self.n_init,
                voting=self.voting
            )

        # Convert back to original label format
        final_labels = self.transformer.decode(y_encoded)
        return data, final_labels

    def _load_raw_data(self):
        all_data = []
        all_labels = []

        # Load class_N_data.npz files
        files = sorted(glob(os.path.join(self.dataset_dir, 'class_*_data.npz')),
                       key=lambda x: int(re.search(r'class_(\d+)_data\.npz', x).group(1)))

        # Validate file count
        if len(files) != len(self.class_labels):
            raise ValueError(f"Found {len(files)} data files, but {len(self.class_labels)} classes are required")

        # Load data for each class
        for file_path in files:
            # Extract class index
            class_num = int(re.search(r'class_(\d+)_data\.npz', file_path).group(1))

            with np.load(file_path) as npz_data:
                class_data = npz_data['embeddings']
                n_samples = class_data.shape[0]
                all_data.append(class_data)
                all_labels.extend([self.class_labels[class_num]] * n_samples)

        return np.vstack(all_data), np.array(all_labels)

class LabelTransformer:
    def __init__(self, class_labels):
        self.class_labels = [tuple(label) for label in class_labels]
        self.label_to_int = {label: idx for idx, label in enumerate(self.class_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}

    def encode(self, labels):
        return np.array([self.label_to_int[tuple(label)] for label in labels])

    def decode(self, encoded_labels):
        return np.array([list(self.int_to_label[label]) for label in encoded_labels])

class StructuredDataSaver:
    def __init__(self, output_dir="Kmer_KPCA1_ENN1_OSS_CC_dataset_20d"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.class_counts = {}  # Sample counter

    def save_by_class(self, data, labels, class_labels):
        y_encoded = LabelTransformer(class_labels).encode(labels)

        for class_id in range(len(class_labels)):
            class_indices = np.where(y_encoded == class_id)[0]
            class_data = data[class_indices]
            self.class_counts[class_id] = len(class_data)  # Record sample count

            # Generate sample names (class_0_sample_0001 format)
            names = [f"class_{class_id}_sample_{i:04d}" for i in range(len(class_data))]

            # Save npz file containing names and embeddings
            filename = f"class_{class_id}_data.npz"
            filepath = os.path.join(self.output_dir, filename)
            np.savez(filepath,
                     names=np.array(names),
                     embeddings=class_data)

            print(f"Class {class_id}: saved {len(class_data)} samples to: {filepath}")

        # Save label mapping
        label_mapping_filepath = os.path.join(self.output_dir, "label_mapping.npz")
        label_mapping = {i: list(l) for i, l in enumerate(class_labels)}
        np.savez(label_mapping_filepath,
                 class_labels=np.array(class_labels, dtype=np.int8),
                 label_mapping=json.dumps(label_mapping))

        # Save statistics
        self._save_statistics()

    def _save_statistics(self):
        """Save statistics to txt and json files"""
        txt_path = os.path.join(self.output_dir, 'class_counts.txt')
        json_path = os.path.join(self.output_dir, 'class_counts.json')

        # Text format
        with open(txt_path, 'w') as f:
            for class_id, count in self.class_counts.items():
                f.write(f"Class {class_id}: {count} samples\n")

        # JSON format
        with open(json_path, 'w') as f:
            json.dump(self.class_counts, f, indent=4)

        print(f"\nSample statistics saved to: {txt_path} and {json_path}")


if __name__ == "__main__":
    CONFIG = {
        'dataset_dir': 'Kmer_KPCA1_ENN1_OSS_dataset_20d_k1_5',
        'class_labels': [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
            [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]
        ],
        'min_samples': 1180,
        'n_init': 10,
        'voting': 'auto'
    }

    # Initialize loader
    loader = DataLoader(
        dataset_dir=CONFIG['dataset_dir'],
        class_labels=CONFIG['class_labels'],
        min_samples=CONFIG['min_samples'],
        n_init=CONFIG['n_init'],
        voting=CONFIG['voting']
    )

    try:
        final_data, final_labels = loader.load_and_process()

        # Save data (using enhanced saver)
        saver = StructuredDataSaver()
        saver.save_by_class(final_data, final_labels, CONFIG['class_labels'])

        # Verify file contents
        test_file = np.load(os.path.join(
            "Kmer_KPCA1_ENN1_OSS_CC_dataset_20d", "class_0_data.npz"))
        print("\nFile field verification:", test_file.files)
        print("Sample name examples:", test_file['names'][:3])
        print("Feature data dimensions:", test_file['embeddings'].shape)

        # Print statistics
        stats_file = os.path.join(
            "Kmer_KPCA1_ENN1_OSS_CC_dataset_20d", "class_counts.txt")
        if os.path.exists(stats_file):
            print("\nSample statistics:")
            with open(stats_file, 'r') as f:
                print(f.read())

    except Exception as e:
        print(f"Processing failed: {str(e)}")
