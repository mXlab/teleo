import numpy as np
import glob

HUGGING_FACE_AUTH_TOKEN=""

# 1. visit hf.co/pyannote/embedding and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=HUGGING_FACE_AUTH_TOKEN)

from pyannote.audio import Inference
inference = Inference(model, window="whole")

# Create data.
filepaths = sorted(glob.glob("./Voices/split/*.wav"))

embeddings = []

for f in filepaths:
  embeddings.append( inference(f) )

embeddings = np.asarray(embeddings)
print( embeddings.shape )

# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

from scipy.spatial.distance import cdist, cosine

inv_dist = 1 - cdist(embeddings, embeddings, metric = 'cosine')
print(inv_dist)

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(inv_dist)
plt.show()

#from scipy.spatial.distance import cdist
#distance = cdist(embedding1, embedding2, metric="cosine")[0,0]
# `distance` is a `float` describing how dissimilar speakers 1 and 2 are.
