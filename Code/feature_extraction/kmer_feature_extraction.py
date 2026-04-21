import numpy as np
from itertools import product
from collections import Counter
import pandas as pd
import os
import chardet


def kmer_feature(protein_sequences, k=3, use_grouping=True, group_method='charge'):
    """
    Compute k-mer features for protein sequences.

    Parameters:
    - protein_sequences: list of protein sequences
    - k: k-mer length, default is 3 (tripeptides)
    - use_grouping: whether to use amino acid grouping for dimensionality reduction;
                    True = grouped reduction, False = full 20 amino acids
    - group_method: grouping method ('chemical', 'hydrophobic', 'charge')

    Returns:
    - K-mer feature matrix (num_sequences x feature_dim)

    Feature dimension description:
    Full mode (use_grouping=False):
    - k=1: 20^1 = 20 dims (single amino acid composition)
    - k=2: 20^2 = 400 dims (dipeptide composition)
    - k=3: 20^3 = 8000 dims (tripeptide composition)
    - k=4: 20^4 = 160000 dims (tetrapeptide composition, very high-dimensional)

    Grouped mode (use_grouping=True):
    chemical grouping (6 groups): k=1->6, k=2->36, k=3->216, k=4->1296
    hydrophobic grouping (3 groups): k=1->3, k=2->9, k=3->27, k=4->81
    charge grouping (4 groups): k=1->4, k=2->16, k=3->64, k=4->256
    """

    def kmer(sequence, k):
        """
        Compute k-mer counts for a sequence
        """
        kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        return Counter(kmers)

    if use_grouping:
        # Use amino acid grouping (reduced-dimensionality version)
        if group_method == 'chemical':
            # Chemical property grouping (6 groups)
            aa_to_group = {
                'A': 'H', 'V': 'H', 'L': 'H', 'I': 'H', 'M': 'H', 'F': 'H', 'W': 'H',  # Hydrophobic
                'S': 'P', 'T': 'P', 'N': 'P', 'Q': 'P', 'Y': 'P', 'C': 'P',  # Polar uncharged
                'K': '+', 'R': '+', 'H': '+',  # Positively charged
                'D': '-', 'E': '-',  # Negatively charged
                'G': 'G',  # Glycine
                'P': 'Pr'  # Proline
            }
            groups = ['H', 'P', '+', '-', 'G', 'Pr']

        elif group_method == 'hydrophobic':
            # Hydrophobicity grouping (3 groups)
            aa_to_group = {
                'A': 'H', 'V': 'H', 'L': 'H', 'I': 'H', 'M': 'H', 'F': 'H', 'W': 'H', 'P': 'H',  # Hydrophobic
                'S': 'P', 'T': 'P', 'N': 'P', 'Q': 'P', 'Y': 'P', 'C': 'P', 'G': 'P',  # Polar
                'K': 'C', 'R': 'C', 'H': 'C', 'D': 'C', 'E': 'C'  # Charged
            }
            groups = ['H', 'P', 'C']

        elif group_method == 'charge':
            # Charge grouping (4 groups)
            aa_to_group = {
                'K': '+', 'R': '+', 'H': '+',  # Positively charged
                'D': '-', 'E': '-',  # Negatively charged
                'S': 'N', 'T': 'N', 'N': 'N', 'Q': 'N', 'Y': 'N', 'C': 'N',  # Polar uncharged
                'A': 'X', 'V': 'X', 'L': 'X', 'I': 'X', 'M': 'X', 'F': 'X', 'W': 'X', 'G': 'X', 'P': 'X'  # Nonpolar
            }
            groups = ['+', '-', 'N', 'X']
        # Generate all possible grouped k-mer combinations
        kmers_set = [''.join(p) for p in product(groups, repeat=k)]
        def group_sequence(sequence):
            """Convert an amino acid sequence to its grouped sequence"""
            return ''.join(aa_to_group.get(aa, 'X') for aa in sequence)
        print(f"K-mer parameter settings:")
        print(f"- k-mer length: {k}")
        print(f"- Use grouping: {use_grouping}")
        print(f"- Grouping method: {group_method}")
        print(f"- Number of groups: {len(groups)}")
        print(f"- Feature dimension: {len(groups)}^{k} = {len(kmers_set)}")
        feature_list = []
        for sequence in protein_sequences:
            # Convert amino acid sequence to grouped sequence
            grouped_seq = group_sequence(sequence)
            # Compute k-mer of grouped sequence
            kmers = kmer(grouped_seq, k)
            # Generate feature vector
            feature_vector = np.array([kmers.get(kmer_item, 0) for kmer_item in kmers_set])
            feature_list.append(feature_vector)

    else:
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
        kmers_set = [''.join(p) for p in product(amino_acids, repeat=k)]
        print(f"K-mer parameter settings:")
        print(f"- k-mer length: {k}")
        print(f"- Use grouping: {use_grouping}")
        print(f"- Amino acid types: 20")
        print(f"- Feature dimension: 20^{k} = {len(kmers_set)}")
        feature_list = []
        for sequence in protein_sequences:
            # Compute k-mer of original sequence
            kmers = kmer(sequence, k)
            # Generate feature vector
            feature_vector = np.array([kmers.get(kmer_item, 0) for kmer_item in kmers_set])
            feature_list.append(feature_vector)
    return np.array(feature_list)

# Read FASTA-format file
def read_fasta_with_labels(file_path):
    """
    Read a FASTA file that contains labels
    """
    sequences = []
    labels = []
    # Detect file encoding
    with open(file_path, 'rb') as raw_file:
        result = chardet.detect(raw_file.read())
    encoding = result['encoding'] or 'utf-8'
    with open(file_path, 'r', encoding=encoding) as file:
        label = None
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    labels.append(label)
                label = 'Positive' if 'Positive' in line else 'Negative'
                sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
            labels.append(label)
    return sequences, labels


def read_fasta_simple(file_path):
    """
    Read a plain FASTA file (no labels)
    """
    sequences = []
    names = []
    # Detect file encoding
    with open(file_path, 'rb') as raw_file:
        result = chardet.detect(raw_file.read())
    encoding = result['encoding'] or 'utf-8'
    with open(file_path, 'r', encoding=encoding) as file:
        name = None
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    names.append(name)
                name = line[1:]  # Remove leading ">"
                sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
            names.append(name)
    return sequences, names


def process_single_file_with_labels(file_path, output_file, k=3, use_grouping=True, group_method='charge'):
    """
    Process a single FASTA file that contains labels
    """
    print(f"Starting to process file: {file_path}")
    # Read sequences and labels
    sequences, labels = read_fasta_with_labels(file_path)
    print(f"Read {len(sequences)} sequences")
    # Extract features
    features = kmer_feature(sequences, k=k, use_grouping=use_grouping, group_method=group_method)
    print(f"Feature extraction complete, feature matrix shape: {features.shape}")
    # Convert to DataFrame
    df = pd.DataFrame(features)
    df['Label'] = labels
    # Output to file
    df.to_csv(output_file, index=False)
    print(f"Feature extraction complete, results saved to {output_file}")


def process_folder(input_dir, output_dir, k=3, use_grouping=True, group_method='charge'):
    """
    Process all FASTA files in a folder
    """
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"K-mer feature extraction parameters:")
    print(f"- k-mer length: {k}")
    print(f"- Use grouping: {use_grouping}")
    if use_grouping:
        print(f"- Grouping method: {group_method}")
    print("-" * 50)
    # Iterate over all files in the input folder
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith((".txt", ".fasta", ".fa")):
            input_path = os.path.join(input_dir, filename)
            # Read sequences (without labels)
            sequences, names = read_fasta_simple(input_path)
            print(f"Processing: {filename}, contains {len(sequences)} sequences")
            # Extract features
            features = kmer_feature(sequences, k=k, use_grouping=use_grouping, group_method=group_method)
            # Save as npz format
            output_name = f"processed_{os.path.splitext(filename)[0]}.npz"
            output_path = os.path.join(output_dir, output_name)
            np.savez(output_path, names=names, embeddings=features)

            print(f"Saved to: {output_path}, feature shape: {features.shape}")
            processed_count += 1

    print(f"Batch processing complete! Processed {processed_count} files in total")

# Main program
if __name__ == "__main__":
    k = 1
    use_grouping = False  # True = grouped reduction, False = full 20 amino acids
    group_method = 'chemical'  # 'chemical', 'hydrophobic', 'charge'

    # Feature dimension description
    print("=" * 60)
    print("Protein K-mer Feature Extraction")
    print("=" * 60)

    if use_grouping:
        dim_map = {
            'chemical': {1: 6, 2: 36, 3: 216, 4: 1296},
            'hydrophobic': {1: 3, 2: 9, 3: 27, 4: 81},
            'charge': {1: 4, 2: 16, 3: 64, 4: 256}
        }
        total_dim = dim_map[group_method][k]
        print(f"Using amino acid grouping: {group_method}")
        print(f"k-mer length: {k}")
        print(f"Feature dimension: {len(dim_map[group_method])}^{k} = {total_dim}")

        group_info = {
            'chemical': '6 groups: Hydrophobic(H), Polar uncharged(P), Positive(+), Negative(-), Glycine(G), Proline(Pr)',
            'hydrophobic': '3 groups: Hydrophobic(H), Polar(P), Charged(C)',
            'charge': '4 groups: Positive(+), Negative(-), Polar uncharged(N), Nonpolar(X)'
        }
        print(f"Grouping description: {group_info[group_method]}")

    else:
        total_dim = 20 ** k
        print(f"Using all 20 amino acids")
        print(f"k-mer length: {k}")
        print(f"Feature dimension: 20^{k} = {total_dim}")
        print(f"Amino acids: ACDEFGHIKLMNPQRSTVWY")

    print("=" * 60)

    # Select processing mode
    print("\nStarting to process training set...")
    input_folder = r"Train dataset"
    output_folder = f"Kmer_FE_Train_dataset_{total_dim}d_k{k}"
    if use_grouping:
        output_folder += f"_{group_method}"
    process_folder(input_folder, output_folder,
                   k=k, use_grouping=use_grouping,
                   group_method=group_method)

    print("\nStarting to process test set...")
    input_folder = r"Test dataset"
    output_folder = f"Kmer_FE_Test_dataset_{total_dim}d_k{k}"
    if use_grouping:
        output_folder += f"_{group_method}"
    process_folder(input_folder, output_folder,
                   k=k, use_grouping=use_grouping,
                   group_method=group_method)

    print("\n" + "=" * 60)
    print("All files processed!")
    print("=" * 60)