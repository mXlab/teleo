import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
import glob

from teleospeaker import *

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Train a simple familiarity identification system.")
parser.add_argument("--dataset-chunk-duration", '-dcd', type=float, default=1.0, help="Dataset chunk duration in seconds.")
parser.add_argument("--dataset-chunk-overlap", '-dco', type=float, default=0.5, help="Dataset chunk overlap as a fraction of chunk duration.")
parser.add_argument("--csv-path", '-csv', type=str, default="./Training/SoundsFilesClasser.csv", help="Path to the CSV file containing file names and labels.")
parser.add_argument("--voices-path", '-vp', type=str, default="./samples", help="Path to the folder containing audio files.")
parser.add_argument("--k-nearest-neighbors", '-k', type=int, default=20, help="Number of nearest neighbors.")
args = parser.parse_args()

# Use parsed arguments
input_dim_val = 512
dataset_chunk_duration_val = args.dataset_chunk_duration
dataset_chunk_overlap_val = args.dataset_chunk_overlap
csv_path = args.csv_path
voices_path = args.voices_path
k_nearest_neighbors = args.k_nearest_neighbors

# Read the CSV file
csv_data = pd.read_csv(csv_path)
print(csv_data.columns)
if 'filename' in csv_data.columns:
    file_paths = [os.path.join(voices_path, file_name) for file_name in csv_data['filename']]
elif 'file_name' in csv_data.columns:
    file_paths = [os.path.join(voices_path, file_name) for file_name in csv_data['file_name']]
else:
    raise KeyError("The column for file names is missing in the CSV file.")
labels = csv_data['label'].tolist()

# Create the datasets
dataset = SpeakerEmbeddingDataset(file_paths, labels=labels, chunk_duration=dataset_chunk_duration_val, chunk_overlap=dataset_chunk_overlap_val)

def get_data(dataset):
  embeddings = []
  targets = []

  for emb, target in dataset:
    embeddings.append(emb.numpy())         # Convert torch.Tensor to NumPy array
    targets.append(target)

  # Convert to final NumPy arrays
  return np.stack(embeddings), np.array(targets)

known_data, known_targets = get_data(dataset)

def process_embedding(embedding):
    similarity, trust, weighted_trust, top_trust = real_time_manager.get_trust_metrics(embedding)
    print(f"Similarity: {similarity}\nTrust: {trust}\nWeighted Trust: {weighted_trust}\nTop Trust: {top_trust}")

real_time_manager = SpeakerRealTimeDataManager(initial_embeddings=known_data, initial_targets=known_targets, k_nearest_neighbors=k_nearest_neighbors)

real_time_tester = SpeakerRealTimeProcessing(callback=process_embedding, duration=1)
real_time_tester.run()

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.heatmap(inv_dist)
# plt.show()
