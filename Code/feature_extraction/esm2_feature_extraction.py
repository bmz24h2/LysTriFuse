from transformers import EsmModel, EsmTokenizer
import torch
import numpy as np
from torch import nn
import os
import chardet
import shutil
import os

# Define function to parse FASTA files
def read_fasta(file_path):
    """
    Read a FASTA file and return a list of [name, sequence] pairs.
    Format: [[name1, seq1], [name2, seq2], ...]
    """

    sequences = []
    with open(file_path, 'rb') as raw_file:
        result = chardet.detect(raw_file.read())
    encoding = result['encoding'] or 'utf-8'  # Default to utf-8 if detection fails
    # enc = detect_encoding(file_path)  #
    with open(file_path, "r", encoding=encoding) as f:
    # with open(file_path, "r", encoding=enc, errors='replace') as f:  # Explicitly specify encoding as UTF-8
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    sequences.append([current_name, "".join(current_seq)])
                current_name = line[1:]  # Remove leading ">"
                current_seq = []
            else:
                current_seq.append(line)
        # Add the last sequence
        if current_name is not None:
            sequences.append([current_name, "".join(current_seq)])
    return sequences

# Load model and tokenizer
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(
    model_name,
    add_pooling_layer=False  # Explicitly disable the unneeded pooling layer
)

# Add a linear layer to reduce model output to 100 dimensions
class ESMWithReduction(nn.Module):
    def __init__(self, original_model, output_dim=100):  # Modify 120 here to your desired dimension
        super(ESMWithReduction, self).__init__()
        self.original_model = original_model
        self.reduction_layer = nn.Linear(original_model.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Output from the original model
        outputs = self.original_model(input_ids=input_ids, attention_mask=attention_mask)
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        # Apply the dimensionality reduction layer
        reduced_output = self.reduction_layer(last_hidden_state)
        return reduced_output

# Create the reduced model
reduced_model = ESMWithReduction(model)
reduced_model.eval()  # Set to evaluation mode

# Define the feature extraction function
def get_esm_embeddings(sequence, max_length=512):
    """
    Input a nucleotide sequence and return the dimensionality-reduced feature vector.
    Sequences longer than max_length will be truncated.
    """
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,      # Automatically pad to the same length
        truncation=True,   # Automatically truncate to the model's maximum length
        max_length=max_length,  # Set maximum length
    )
    # Forward pass
    with torch.no_grad():
        outputs = reduced_model(**inputs)

    print(f"[DEBUG] Input sequence length: {len(sequence)}, "
          f"token count: {inputs['input_ids'].shape[1]}, "
          f"reduced output shape: {outputs.shape}")  # (batch, seq_len, 100)
    # Mean pooling to get global features
    embeddings = outputs.mean(dim=1).squeeze().numpy()
    return embeddings

# 4. Main pipeline: read files, extract features, save results
def main(input_file, output_file):
    # Read FASTA file
    sequences = read_fasta(input_file)  # Read all sequences
    print(f"Read {len(sequences)} sequences from {input_file}")
    # Extract features and save
    features = []
    names = []
    ''' name '''''' name '''''' name '''''' name '''''' name '''''' name '''''' name '''
    for name, seq in sequences:  # Process each sequence individually
        try:
            embedding = get_esm_embeddings(seq)  # Process this sequence
            print(f"Sample: {name} | Sequence length: {len(seq)} | embedding shape: {embedding.shape}")
            features.append(embedding)
            names.append(name)
           # print(f"Processed sequence: {name}")
        except Exception as e:
            print(f"Failed to process sequence {name}: {str(e)}")
    # Convert to NumPy array
    features_array = np.array(features)
    # Save as .npz file (names and feature vectors)
    np.savez(output_file, names=names, embeddings=features_array)
    print(f"Dimensionality-reduced feature vectors saved to {output_file}")

# Process text files in the specified input folder and save results to the specified output folder
def process_folder(input_dir, output_dir):
    # Create output folder (if it does not exist)
    os.makedirs(output_dir, exist_ok=True)
    # Iterate over all files in the input folder
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):  # Process text files only
            # Build full paths
            input_path = os.path.join(input_dir, filename)
            output_name = f"processed_{os.path.splitext(filename)[0]}.npz"
            output_path = os.path.join(output_dir, output_name)
            # Call the original processing logic
            main(input_path, output_path)

# Run the script
if __name__ == "__main__":
    '''Feature extraction for the Test dataset'''
    # input_folder = r"Test dataset"    # Input FASTA file path
    # output_folder = r"ESM2_FE_Test_dataset_max_512_100d"  # Output filename

    input_folder = r"Train dataset"    # Input FASTA file path
    output_folder = (r"ESM2_FE_Train_dataset_max_512_100d")  # Output filename
    process_folder(input_folder, output_folder)