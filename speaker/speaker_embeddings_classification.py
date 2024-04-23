import malaya_speech
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob

from sklearn.model_selection import train_test_split

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Create data.
speakers = ['1-Victor', '2-Sofian', '3-Etienne', '4-Natalia']
# Type of model.
regression_model = False
# Classification/regression model filename.
model_filename = 'trustnet.pt'

class Data(Dataset):
  def __init__(self, X_train, y_train):
    self.X = torch.from_numpy(X_train.astype(np.float32))
    if regression_model:
      self.y = torch.from_numpy(y_train).type(torch.FloatTensor)
    else:
      self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
  def __len__(self):
    return self.len
  
# number of features (len of X cols)
input_dim = 512
# number of hidden neurons
hidden_dim = 32
hidden_dim2 = 32
# number of classes (unique of y)
n_classes = len(speakers)

def load_wav(file):
  return malaya_speech.load(file)[0]

# Speaker vector model.
speaker_encoder = malaya_speech.speaker_vector.deep_model('vggvox-v2')

vad_model = malaya_speech.vad.deep_model(model='vggvox-v2', quantized=True)

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

p = malaya_speech.Pipeline()
p.batching(8000).map(speaker_encoder)

X = np.empty((0, 512))
Y = np.array([])
for i in range(len(speakers)):
  s = speakers[i]
  filepath = "./Voices/VOIX TELEO " + s + ".wav"
  x_speaker = p.emit(load_wav(filepath))['speaker-vector']
  y_speaker = np.full((x_speaker.shape[0]), i)
  X = np.vstack([X, x_speaker])
  Y = np.append(Y, y_speaker)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print(X_test)
print(Y_test)

traindata = Data(X_train, Y_train)
batch_size = 4
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)

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
  for i, data in enumerate(trainloader, 0):
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


testdata = Data(X_test, Y_test)
testloader = DataLoader(testdata, batch_size=batch_size, 
                        shuffle=True, num_workers=2)

correct, total = 0, 0
# no need to calculate gradients during inference
with torch.no_grad():
  for data in testloader:
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

print(f'Accuracy of the network on the {len(testdata)} test data: {100 * correct // total} %')


# Save trustnet model.
print("Saving model")
torch.save(trustnet.state_dict(), model_filename)

# # print(r['speaker-vector'])

# ##print(model(load_wav(filepaths[0])))

# # calculate similarity
# from scipy.spatial.distance import cdist, cosine

# # print(type(r['speaker-vector']))
# print(1 - cdist(r['speaker-vector'], r['speaker-vector'], metric = 'cosine'))