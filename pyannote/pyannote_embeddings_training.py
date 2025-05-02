import numpy as np
import librosa
import webrtcvad
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from pyannote.audio import Model, Inference
import numpy as np
import glob

from teleospeaker import SpeakerEmbeddingDataset


# Create data.
speakers = ['1-Victor', '2-Sofian', '3-Etienne', '4-Natalia']

# Type of model.
regression_model = False

# Classification/regression model filename.
model_filename = 'trustnet.pt'

# number of features (len of X cols)
input_dim = 512
# number of hidden neurons
hidden_dim = 32
hidden_dim2 = 16
# number of classes (unique of y)
n_classes = len(speakers)

labels = range(n_classes)

# Create a classifier model with one hidden layer with RELU activation, for classifying the speaker vectors.
if regression_model:
  trustnet = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim2),
    nn.Sigmoid(),
    nn.Linear(hidden_dim2, 1))
else:
  trustnet = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, n_classes),
    nn.Sigmoid())

# Usage
file_paths = [ "./Voices/VOIX TELEO " + s + ".wav" for s in speakers ]
dataset = SpeakerEmbeddingDataset(file_paths, labels=labels, chunk_duration=1, chunk_overlap=0.95)

# Assuming 'SpeakerEmbeddingDataset' has been defined and instantiated as 'dataset'
# Define the sizes for your train and test sets
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% of the dataset for training
test_size = total_size - train_size  # The remainder for testing

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders if you want to iterate over batches
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

if regression_model:
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(trustnet.parameters(), lr=0.1)
  # optimizer = torch.optim.Adam(trustnet.parameters(), lr=0.1)
else:
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(trustnet.parameters(), lr=0.1)

epochs = 100
for epoch in range(epochs):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    # set optimizer to zero grad to remove previous epoch gradients
    optimizer.zero_grad()
    # forward propagation
    outputs = trustnet(inputs)
    loss = criterion(outputs, labels)
    # backward propagation
    loss.backward()
    # optimize
    optimizer.step()
    running_loss += loss.item()
  
  # display statistics
  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')


correct, total = 0, 0
# no need to calculate gradients during inference
with torch.no_grad():
  for data in test_loader:
    inputs, labels = data
    # calculate output by running through the network
    outputs = trustnet(inputs)
    # get the predictions
    if regression_model:
      predicted = torch.round(outputs.data).flatten()
      correct += labels.size(0) - np.abs(predicted - labels).sum().item() / n_classes
    else:
      __, predicted = torch.max(outputs.data, 1)
      # update results., 2.]) te
      correct += (predicted == labels).sum().item()
    total += labels.size(0)
    
    # print(inputs, labels, outputs, outputs.data, predicted, correct, total, predicted == labels, (predicted == labels).sum(), np.abs(predicted-labels), np.abs(predicted-labels).sum())

print(f'Accuracy of the network on the {len(test_dataset)} test data: {100 * correct // total} %')


# Save trustnet model.
print("Saving model")
torch.save(trustnet, model_filename)
# torch.save(trustnet.state_dict(), model_filename)
