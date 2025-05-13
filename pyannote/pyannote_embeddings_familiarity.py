import argparse
import signal
import time
import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

from teleospeaker import *
from messaging import *

def interrupt(signup, frame):
    global client, server
    print("Exiting program...")
    osc_terminate()
    sys.exit()

signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, interrupt)

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Train a simple familiarity identification system.")
parser.add_argument("--dataset-chunk-duration", '-dcd', type=float, default=1.0, help="Dataset chunk duration in seconds.")
parser.add_argument("--dataset-chunk-overlap", '-dco', type=float, default=0.5, help="Dataset chunk overlap as a fraction of chunk duration.")
parser.add_argument("--sample-duration", '-sd', type=float, default=3.0, help="Recording sample duration in seconds.")
parser.add_argument("--csv-path", '-csv', type=str, default="./Training/SoundsFilesClasser.csv", help="Path to the CSV file containing file names and labels.")
parser.add_argument("--voices-path", '-vp', type=str, default="./samples", help="Path to the folder containing audio files.")
parser.add_argument("--k-nearest-neighbors", '-k', type=int, default=20, help="Number of nearest neighbors.")
parser.add_argument("--osc-ip", '-oip', type=str, default="192.168.0.255", help="OSC remote IP.")
parser.add_argument("--osc-send-port", '-osp', type=int, default=8000, help="OSC send port")
parser.add_argument("--osc-receive-port", '-orp', type=int, default=7070, help="OSC receive port")
parser.add_argument("--pyannote-model-path", '-pmp', type=str, default="pyannote/embedding", help="Model path for pyannote")
parser.add_argument("--hugging-face-auth-token", "-hf", type=str, default="", help="Hugging Face authentication token")

args = parser.parse_args()

# Use parsed arguments
input_dim_val = 512
dataset_chunk_duration_val = args.dataset_chunk_duration
dataset_chunk_overlap_val = args.dataset_chunk_overlap
sample_duration = args.sample_duration
csv_path = args.csv_path
voices_path = args.voices_path
k_nearest_neighbors = args.k_nearest_neighbors
model_path = args.pyannote_model_path
hugging_face_auth_token = args.hugging_face_auth_token

# OSC setup
print("Starting OSC")
osc_startup()

def receive_reward(reward):
  print(f"Received reward: {reward} -----------------------------------------")
  real_time_manager.add_current_embedding(reward)

osc = OscHelper("main", args.osc_ip, args.osc_send_port, args.osc_receive_port)
osc.map("/reward", receive_reward)

# Read the CSV file
print("Loading data")
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
print("Dataset creation")
dataset = SpeakerEmbeddingDataset(file_paths, labels=labels, chunk_duration=dataset_chunk_duration_val, chunk_overlap=dataset_chunk_overlap_val, model_path=model_path, hugging_face_auth_token=hugging_face_auth_token)

# Extract embeddings and targets from datasest.
def get_data(dataset):
  embeddings = []
  targets = []

  for emb, target in dataset:
    embeddings.append(emb.numpy())         # Convert torch.Tensor to NumPy array
    targets.append(target)

  # Convert to final NumPy arrays
  return np.stack(embeddings), np.array(targets)

known_data, known_targets = get_data(dataset)

# Callback for real-time processing of new embedding.
def process_embedding(embedding):
    # Compute metrics.
    similarity, trust, weighted_trust, top_trust = real_time_manager.get_trust_metrics(embedding)
    real_time_manager.register_current_sample(embedding)

    # Send metrics.
    osc.send_bundle(
       {
        "/new-sample" : [],
        "/similarity": similarity.tolist(),
        "/mean-trust": trust.tolist(),
        "/weighted-trust": weighted_trust.tolist(),
        "/top-trust": top_trust.tolist(),
       }
    )
    print(f"Similarity: {similarity}\nTrust: {trust}\nWeighted Trust: {weighted_trust}\nTop Trust: {top_trust}")

print("Starting main program")
real_time_manager = SpeakerRealTimeDataManager(initial_embeddings=known_data, initial_targets=known_targets, k_nearest_neighbors=k_nearest_neighbors)

real_time_tester = SpeakerRealTimeProcessing(callback=process_embedding, duration=sample_duration, model_path=model_path, hugging_face_auth_token=hugging_face_auth_token)
real_time_tester.run(callback=osc_process)
