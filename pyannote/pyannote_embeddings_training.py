import argparse
import pandas as pd
import numpy as np
import librosa
import webrtcvad
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from pyannote.audio import Model, Inference
import os

from teleospeaker import SpeakerEmbeddingDataset

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Train a speaker embedding model.")
parser.add_argument("--regression", '--r', action='store_true', help="Use regression model instead of classification.")
parser.add_argument("--input_dim", '--id', type=int, default=512, help="Input dimension of the model.")
parser.add_argument("--hidden_dim", '--hd', type=int, default=32, help="Number of neurons in the first hidden layer.")
parser.add_argument("--hidden_dim2", '--hd2', type=int, default=16, help="Number of neurons in the second hidden layer.")
parser.add_argument("--epochs", '--e', type=int, default=10, help="Number of training epochs.")
parser.add_argument("--chunk-duration", '--cd', type=float, default=1.0, help="Chunk duration in seconds.")
parser.add_argument("--chunk-overlap", '--co', type=float, default=0.0, help="Chunk overlap as a fraction of chunk duration.")
parser.add_argument("--batch-size", '--bz', type=int, default=10, help="Batch size for training.")
parser.add_argument("--learning-rate", '--lr', type=float, default=0.1, help="Learning rate for the optimizer.")
parser.add_argument("--dataset-chunk-duration", '--dcd', type=float, default=1.0, help="Dataset chunk duration in seconds.")
parser.add_argument("--dataset-chunk-overlap", '--dco', type=float, default=0.95, help="Dataset chunk overlap as a fraction of chunk duration.")
parser.add_argument("--csv-path", '--csv', type=str, default="./Training/SoundsFilesClasser.csv", help="Path to the CSV file containing file names and labels.")
parser.add_argument("--voices-path", '--vp', type=str, default="./samples", help="Path to the folder containing audio files.")
args = parser.parse_args()

# Use parsed arguments
regression_model = args.regression
input_dim_val = args.input_dim
hidden_dim_val = args.hidden_dim
hidden_dim2_val = args.hidden_dim2
epochs_val = args.epochs
chunk_duration_val = args.chunk_duration
chunk_overlap_val = args.chunk_overlap
batch_size_val = args.batch_size
learning_rate_val = args.learning_rate
dataset_chunk_duration_val = args.dataset_chunk_duration
dataset_chunk_overlap_val = args.dataset_chunk_overlap
csv_path = args.csv_path
voices_path = args.voices_path

# Set the path to the Hugging Face authentication token
HUGGING_FACE_AUTH_TOKEN = ""

# Classification/regression model filename
model_filename = 'trustnet.pt'

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

# Create the dataset
dataset = SpeakerEmbeddingDataset(file_paths, labels, chunk_duration=dataset_chunk_duration_val, chunk_overlap=dataset_chunk_overlap_val, regression_model=regression_model)

# Number of classes (unique labels)
n_classes = len(set(labels))

# Create a classifier or regression model
if regression_model:
    trustnet = nn.Sequential(
        nn.Linear(input_dim_val, hidden_dim_val),
        nn.ReLU(),
        nn.Linear(hidden_dim_val, hidden_dim2_val),
        nn.Sigmoid(),
        nn.Linear(hidden_dim2_val, 1)
    )
else:
    trustnet = nn.Sequential(
        nn.Linear(input_dim_val, hidden_dim_val),
        nn.ReLU(),
        nn.Linear(hidden_dim_val, hidden_dim2_val),
        nn.ReLU(),
        nn.Linear(hidden_dim2_val, n_classes),
        nn.Sigmoid()
    )

# Split the dataset
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% of the dataset for training
test_size = total_size - train_size  # The remainder for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False)

# Define loss and optimizer
if regression_model:
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(trustnet.parameters(), lr=learning_rate_val)

# Training loop
for epoch in range(epochs_val):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # Zero gradients
        optimizer.zero_grad()
        outputs = trustnet(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item()
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')

# Evaluate the model
correct, total = 0, 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data

        outputs = trustnet(inputs)
        if regression_model:
            predicted = torch.round(outputs.data).flatten()
            correct += labels.size(0) - np.abs(predicted - labels).sum().item() / n_classes
        else:
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        total += labels.size(0)
print(f'Accuracy of the network on the {len(test_dataset)} test data: {100 * correct // total} %')

# Save the model
print("Saving model")
torch.save(trustnet.state_dict(), model_filename)