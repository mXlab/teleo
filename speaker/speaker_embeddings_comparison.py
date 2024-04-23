import malaya_speech
import sounddevice

model = malaya_speech.speaker_vector.deep_model('vggvox-v2')

# vad_model = malaya_speech.vad.webrtc()
vad_model = malaya_speech.vad.deep_model(model = 'vggvox-v2', quantized = True)

from glob import glob
import sklearn.pipeline

# pipeline
def load_wav(file):
  return malaya_speech.load(file)[0]

#speakers = ['1-Victor', '2-Sofian', '3-Etienne', '4-Natalia']
#filepaths = list(map(lambda s : "./Voices/VOIX TELEO " + s + ".wav", speakers))

import glob
filepaths = sorted(glob.glob("./Voices/split/*.wav"))
print(filepaths)

p = malaya_speech.Pipeline()
p.foreach_map(load_wav).map(model)

r = p.emit(filepaths)
# print(r['speaker-vector'])

##print(model(load_wav(filepaths[0])))

# calculate similarity
from scipy.spatial.distance import cdist, cosine

inv_dist = 1 - cdist(r['speaker-vector'], r['speaker-vector'], metric = 'cosine')
print(inv_dist)

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(inv_dist)
plt.show()