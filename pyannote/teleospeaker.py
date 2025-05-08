import numpy as np
import librosa
import webrtcvad
import torch
from torch.utils.data import Dataset
from pyannote.audio import Model, Inference
import sounddevice as sd
import time
from queue import Queue
import os
import wave
import csv
import random
from scipy.spatial.distance import cdist

# Set the path to the Hugging Face authentication token
HUGGING_FACE_AUTH_TOKEN = ""

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, file_paths, labels=False, sample_rate=48000, chunk_duration=1, chunk_overlap=0, regression_model=False):
        super(SpeakerEmbeddingDataset, self).__init__()
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.regression_model = regression_model
        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGING_FACE_AUTH_TOKEN)
        self.inference = Inference(self.model, window="whole")
        self.vad = webrtcvad.Vad(3)
        if self.labels:
            self.embeddings, self.targets = self.preprocess_files()
        else:
            self.embeddings = self.preprocess_files()

    def preprocess_files(self):
        embeddings = []
        if self.labels:
            targets = []
        for i, file_path in enumerate(self.file_paths):
            audio, __ = librosa.load(file_path, sr=self.sample_rate)
            voiced_audio = self.apply_vad(audio)
            chunks = self.extract_chunks(voiced_audio)
            for chunk in chunks:
                tensor_chunk = torch.tensor(chunk).unsqueeze(0).float()
                embedding = self.inference({'waveform': tensor_chunk, 'sample_rate': self.sample_rate})
                embeddings.append(embedding)
                if self.labels:
                    targets.append(self.labels[i])  # Use the label from the CSV
        if self.labels:
            targets = np.asarray(targets)
            if self.regression_model:
                targets = torch.from_numpy(targets).type(torch.FloatTensor)
            else:
                targets = torch.from_numpy(targets).type(torch.LongTensor)
            return embeddings, targets
        else:
            return embeddings

    def apply_vad(self, audio):
        frame_duration = 0.02  # 20 ms
        frame_samples = int(self.sample_rate * frame_duration)
        voiced_frames = []
        for i in range(0, len(audio) - frame_samples + 1, frame_samples):
            frame = audio[i:i+frame_samples]
            if self.vad.is_speech((frame * 32767).astype(np.int16).tobytes(), self.sample_rate):
                voiced_frames.append(frame)
        return np.concatenate(voiced_frames)

    def extract_chunks(self, audio):
        chunk_samples = int(self.sample_rate * self.chunk_duration)
        chunk_samples_per_step = int(chunk_samples * (1 - self.chunk_overlap))
        return [audio[i:i + chunk_samples] for i in range(0, len(audio) - chunk_samples + 1, chunk_samples_per_step)]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.labels:
            return torch.tensor(self.embeddings[idx]), self.targets[idx]
        else:
            return torch.tensor(self.embeddings[idx])

class SpeakerRealTimeProcessing:
    def __init__(self, sample_rate=48000, duration=3, frame_duration=0.02, vad_mode=3, callback=None):
        self.sample_rate = sample_rate
        self.duration = duration
        self.frame_duration = frame_duration
        self.frame_samples = int(sample_rate * frame_duration)
        self.channels = 1

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)

        self.callback = callback

        self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGING_FACE_AUTH_TOKEN)
        self.inference = Inference(self.model, window="whole")

        self.audio_queue = Queue()
        self.audio_buffer = np.array([])

        # Lazy assignment for stream so it can be stopped from outside
        self.stream = None

    # Modify the save_audio_to_file method to update the CSV file
    def save_audio_to_file(self, filename, audio_data):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        # Add the new file to the CSV with a random label
        csv_file_path = "Training/SoundsFilesClasser.csv"
        label = random.randint(0, 2)  # Generate a random label between 0 and 2
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([os.path.basename(filename), label])

    # Modify the audio_callback method to save audio
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        if frames == self.frame_samples:
            audio_frame = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            if self.vad.is_speech(audio_frame, self.sample_rate):
                # Append incoming data to audio buffer.
                self.audio_buffer = np.append(self.audio_buffer, indata[:, 0])

                # If audio buffer is full, put it in the queue and save to file.
                if len(self.audio_buffer) >= self.sample_rate * self.duration:
                    # Put audio buffer in queue.
                    self.audio_queue.put(self.audio_buffer[:self.sample_rate * self.duration])
                    # Save to file.
                    filename = os.path.join("samples", f"recording_{int(time.time() * 1000)}.wav")
                    self.save_audio_to_file(filename, self.audio_buffer[:self.sample_rate * self.duration])
                    # Reset buffer.
                    self.audio_buffer = self.audio_buffer[self.sample_rate * self.duration:]

    def process_audio(self, data):
        tensor_data = torch.tensor(data).unsqueeze(0).float()
        x = self.inference({'waveform': tensor_data, 'sample_rate': self.sample_rate})
        if self.callback:
            self.callback(x)
    
    def run(self):
        print("Recording... Press Ctrl+C to stop.")
        with sd.InputStream(samplerate=self.sample_rate,
                            channels=self.channels,
                            callback=self.audio_callback,
                            blocksize=self.frame_samples):
            try:
                while True:
                    if not self.audio_queue.empty():
                        data = self.audio_queue.get()
                        self.process_audio(data)
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Stopped recording.")

# Define the trust function based on label
def trust(label):
    return label - 1

def remapTo01(x, from_min, from_max):
    return (x - from_min) / (from_max - from_min)

# Returns the similarity between two embeddings. Parameter y can be a list of embeddings.
def similarity(x, y):
    if x.ndim == 1: # Check if the embedding is 1D
        x = x.reshape(1, -1)  # Reshape if necessary

    return np.maximum(1 - cdist(x, y, metric='cosine').flatten(), 0)

# Returns a tuple of metrics: similarity, trust, weighted trust, top trust, similarity min, similarity max
# Parameters:
# - x is the embedding to compare
# - embeddings is a list of known embeddings
# - labels is a list of labels corresponding to embeddings
# - k is the number of neighbors
def compute_trust_metrics(x, embeddings, labels, k=20, similarity_min=0, similarity_max=1):
    # Compute similarity
    similarities = similarity(x, embeddings)
    similarity_min = min(np.min(similarities), similarity_min)
    similarity_max = max(np.max(similarities), similarity_max)
    similarities = remapTo01(similarities, similarity_min, similarity_max)

    # Get the indices of the top-k most similar embeddings
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Get the label of the top-k neighbors
    top_label = np.bincount(labels[top_k_indices]).argmax()

    # Compute metrics: average similarity, average trust, weighted average trust
    sum_similarities = 0.0
    sum_trust = 0.0
    sum_weighted_trust = 0.0
    for i in range(k):
        # Compute sums
        sim = similarities[top_k_indices[i]]
        tr = trust(labels[top_k_indices[i]])
        w_tr = sim * tr

        sum_similarities += sim
        sum_trust += tr
        sum_weighted_trust += w_tr
    
    # Calculate metrics
    sim = sum_similarities / k
    tr = sum_trust / k
    w_tr = sum_weighted_trust / sum_similarities
    top_tr = trust(top_label)

    # Return metrics
    return sim, tr, w_tr, top_tr, similarity_min, similarity_max

if __name__ == "__main__":
    processor = SpeakerRealTimeProcessing()
    processor.run()