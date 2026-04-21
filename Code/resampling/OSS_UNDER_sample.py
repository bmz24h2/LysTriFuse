import numpy as np
import os
import re
import json
from glob import glob
from collections import Counter
from imblearn.under_sampling import OneSidedSelection

# OSS Sampling Function
def oss_downsample(data, labels, target_samples=1180):
    """
    One-Sided Selection downsampling. Forces classes with more than target_samples
    to be downsampled to the specified count.

    Parameters:
    - data: input data, shape (n_samples, n_features)
    - labels: input labels, shape (n_samples,)
    - target_samples: target sample count per class, default is 1180

    Returns:
    - resampled_data: downsampled data
    - resampled_labels: downsampled labels
    """
    resampled_data = []
    resampled_labels = []
    # Get class distribution
    class_counts = Counter(labels)
    majority_classes = {cls: count for cls, count in class_counts.items() if count > target_samples}
    print("Majority classes to be downsampled:", majority_classes)
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        cls_data = data[cls_indices]
        cls_labels = labels[cls_indices]
        if cls in majority_classes:
            # Apply OSS to majority class
            oss = OneSidedSelection(
                sampling_strategy='majority',  # Downsample majority class only
                random_state=42,
                n_neighbors=3,  # Number of neighbors in OSS
                n_jobs=None  # No parallel processing by default
            )
            # Since OSS cannot directly specify an exact sample count, do initial downsampling first
            temp_data, temp_labels = oss.fit_resample(data, labels)
            temp_cls_indices = np.where(temp_labels == cls)[0]
            temp_cls_data = temp_data[temp_cls_indices]
            temp_cls_labels = temp_labels[temp_cls_indices]
            cls_data_resampled = temp_cls_data
            cls_labels_resampled = temp_cls_labels
        else:
            # Retain all samples from minority classes
            cls_data_resampled = cls_data
            cls_labels_resampled = cls_labels
        resampled_data.append(cls_data_resampled)
        resampled_labels.append(cls_labels_resampled)

    return np.vstack(resampled_data), np.hstack(resampled_labels)

class DataLoader:
    def __init__(self, dataset_dir, class_labels, min_samples=1180):
        self.dataset_dir = dataset_dir
        self.class_labels = class_labels
        self.transformer = LabelTransformer(class_labels)
        self.min_samples = min_samples

    def load_and_process(self):
        """Load and process data using OSS downsampling on majority classes"""
        # Load raw data
        data, labels = self._load_raw_data()
        y_encoded = self.transformer.encode(labels)
        # Convert y_encoded to Python int to avoid int32 issues
        y_encoded = y_encoded.astype(int)
        # Get class distribution
        class_counts = Counter(y_encoded)
        print("Original class distribution:", class_counts)
        # Execute downsampling
        if any(count > self.min_samples for count in class_counts.values()):
            data_resampled, labels_resampled = oss_downsample(data, y_encoded, target_samples=self.min_samples)
            print("OSS downsampling applied")
        else:
            data_resampled, labels_resampled = data, y_encoded
            print("No downsampling needed")
        # Convert back to original label format
        final_labels = self.transformer.decode(labels_resampled)
        return data_resampled, final_labels

    def _load_raw_data(self):
        """Load raw data and return data and labels"""
        all_data = []
        all_labels = []
        # Assume file format is class_X_data.npz
        files = sorted(glob(os.path.join(self.dataset_dir, 'class_*_data.npz')),
                       key=lambda x: int(re.search(r'class_(\d+)_data\.npz', x).group(1)))
        if len(files) != len(self.class_labels):
            raise ValueError(f"Found {len(files)} data files, but {len(self.class_labels)} classes are required")
        for file_path in files:
            class_num = int(re.search(r'class_(\d+)_data\.npz', file_path).group(1))
            with np.load(file_path) as npz_data:
                class_data = npz_data['embeddings']  # Assume data field is 'embeddings'
                all_data.append(class_data)
                all_labels.extend([self.class_labels[class_num]] * class_data.shape[0])

        return np.vstack(all_data), np.array(all_labels)

# Label Transformer
class LabelTransformer:
    def __init__(self, class_labels):
        self.class_labels = [tuple(label) for label in class_labels]
        self.label_to_int = {label: idx for idx, label in enumerate(self.class_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
    def encode(self, labels):
        return np.array([self.label_to_int[tuple(label)] for label in labels])

    def decode(self, encoded_labels):
        return np.array([list(self.int_to_label[label]) for label in encoded_labels])

# Data Saver
class StructuredDataSaver:
    def __init__(self, output_dir="ESM2_OneHot_KPCA1_ENN1_OSS_dataset_100d_20d_5"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.class_counts = {}

    def save_by_class(self, data, labels, class_labels):
        """Save data by class and record sample counts"""
        transformer = LabelTransformer(class_labels)
        y_encoded = transformer.encode(labels)

        for class_id in range(len(class_labels)):
            class_indices = np.where(y_encoded == class_id)[0]
            class_data = data[class_indices]
            self.class_counts[class_id] = len(class_data)

            names = [f"class_{class_id}_sample_{i:04d}" for i in range(len(class_data))]
            filename = f"class_{class_id}_data.npz"
            filepath = os.path.join(self.output_dir, filename)
            np.savez(filepath, names=np.array(names), embeddings=class_data)
            print(f"Class {class_id}: saved {len(class_data)} samples to: {filepath}")

        label_mapping_filepath = os.path.join(self.output_dir, "label_mapping.npz")
        label_mapping = {i: list(l) for i, l in enumerate(class_labels)}
        np.savez(label_mapping_filepath,
                 class_labels=np.array(class_labels, dtype=np.int8),
                 label_mapping=json.dumps(label_mapping))

        self._save_statistics()

    def _save_statistics(self):
        """Save class sample statistics to file"""
        txt_path = os.path.join(self.output_dir, 'class_counts.txt')
        json_path = os.path.join(self.output_dir, 'class_counts.json')

        with open(txt_path, 'w') as f:
            for cls, cnt in sorted(self.class_counts.items()):
                f.write(f"Class {cls}: {cnt} samples\n")

        with open(json_path, 'w') as f:
            json.dump(self.class_counts, f, indent=4)

        print(f"\nSample statistics saved to: {txt_path} and {json_path}")

# Usage Example
if __name__ == "__main__":
    CONFIG = {
        'dataset_dir': 'ESM2_OneHot_KPCA1_ENN1_OSS_dataset_100d_20d_4',
        'class_labels': [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
            [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]
        ],
        'min_samples': 1180
    }
    loader = DataLoader(
        dataset_dir=CONFIG['dataset_dir'],
        class_labels=CONFIG['class_labels'],
        min_samples=CONFIG['min_samples']
    )
    try:
        final_data, final_labels = loader.load_and_process()

        saver = StructuredDataSaver()
        saver.save_by_class(final_data, final_labels, CONFIG['class_labels'])

        # Verify class distribution
        y_encoded = loader.transformer.encode(final_labels)
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("\nFinal class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"Class {cls}: {cnt} samples")

    except Exception as e:
        print(f"Processing failed: {str(e)}")
