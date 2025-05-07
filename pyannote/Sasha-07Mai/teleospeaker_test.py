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
            is_speech = self.vad.is_speech(audio_frame, self.sample_rate)
            if is_speech:
                self.audio_buffer = np.append(self.audio_buffer, indata[:, 0])

                if len(self.audio_buffer) >= self.sample_rate * self.duration:
                    self.audio_queue.put(self.audio_buffer[:self.sample_rate * self.duration])
                    filename = os.path.join("samples", f"recording_{int(time.time() * 1000)}.wav")
                    self.save_audio_to_file(filename, self.audio_buffer[:self.sample_rate * self.duration])
                    self.audio_buffer = self.audio_buffer[self.sample_rate * self.duration:]

    def process_audio(self, data):
        tensor_data = torch.tensor(data).unsqueeze(0).float()
        embedding = self.inference({'waveform': tensor_data, 'sample_rate': self.sample_rate})
        self.callback(embedding)

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
if __name__ == "__main__":
    processor = SpeakerRealTimeProcessing()
    processor.run()