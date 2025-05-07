import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
import glob
import wave
import sounddevice as sd
import random
import csv

from teleospeaker import *
from pythonosc.udp_client import SimpleUDPClient

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Train a simple familiarity identification system.")
parser.add_argument("--dataset-chunk-duration", '-dcd', type=float, default=5.0, help="Dataset chunk duration in seconds.")
parser.add_argument("--dataset-chunk-overlap", '-dco', type=float, default=0.5, help="Dataset chunk overlap as a fraction of chunk duration.")
parser.add_argument("--voices-path", '-vp', type=str, default="./samples", help="Path to the folder containing audio files.")
parser.add_argument("--k-nearest-neighbors", '-k', type=int, default=20, help="Number of nearest neighbors.")
args = parser.parse_args()

# Use parsed arguments
input_dim_val = 512
dataset_chunk_duration_val = args.dataset_chunk_duration
dataset_chunk_overlap_val = args.dataset_chunk_overlap
voices_path = args.voices_path
k_nearest_neighbors = args.k_nearest_neighbors

known_speakers = ['Sofian', 'Sasha', 'Natalia', 'Alex', 'Vincent']
unknown_speakers = ['Etienne', 'Marianne']

def get_file_paths_and_labels(speakers):
  return [f for s in speakers for f in glob.glob(os.path.join(voices_path, f"*{s}*.wav"))]

known_speakers_file_paths = get_file_paths_and_labels(known_speakers)
unknown_speakers_file_paths = get_file_paths_and_labels(unknown_speakers)

# Create the datasets
known_dataset = SpeakerEmbeddingDataset(known_speakers_file_paths, chunk_duration=dataset_chunk_duration_val, chunk_overlap=dataset_chunk_overlap_val)
unknown_dataset = SpeakerEmbeddingDataset(unknown_speakers_file_paths, chunk_duration=dataset_chunk_duration_val, chunk_overlap=dataset_chunk_overlap_val)

def get_data(dataset):    
  # Assuming each item is a tensor (or tuple of tensors), like (x,) or (x, y)
  data = [item[0].numpy() if isinstance(item, tuple) else item.numpy() for item in dataset]
  return np.stack(data)

known_data = get_data(known_dataset)
unknown_data = get_data(unknown_dataset)

# Compute distances
def familiarity(test, known):

  similarity = 1 - cdist(test, known, metric = 'cosine')

  # Sort distances ascendingly along each row
  nearest_similarity = np.sort(similarity, axis=1)[:, -k_nearest_neighbors:]  # take k smallest distances
  # Compute average for each row
  avg_nearest_similarity = np.mean(nearest_similarity, axis=1)
  # avg_nearest_similarity = np.max(similarity)

  return avg_nearest_similarity

# Initialize the OSC client
osc_ip = "192.168.0.100"
osc_port = 7000
osc_client = SimpleUDPClient(osc_ip, osc_port)

familiarity_scores = None  # Initialize as None

def assess_familiarity(embedding):
    global known_data, familiarity_scores
    embedding = np.atleast_2d(embedding)
    familiarity_scores = familiarity(embedding, known_data)
    known_data = np.vstack([known_data, embedding])
    print(familiarity_scores)

    # Send the familiarity score via OSC
    osc_client.send_message("/familiarity_score", familiarity_scores.tolist())
    print(f"OSC message sent to {osc_ip}:{osc_port}")

def get_familiarity_scores():
    return familiarity_scores

familiarity_scores = familiarity(unknown_data, known_data)
print(familiarity_scores)
print(w_trust)

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.heatmap(familiarity_scores)
# plt.show()

real_time_tester = SpeakerRealTimeProcessing(callback=assess_familiarity)
real_time_tester.run()

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.heatmap(inv_dist)
# plt.show()


