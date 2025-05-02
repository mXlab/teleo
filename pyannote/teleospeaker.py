import numpy as np
import librosa
import webrtcvad
import torch
from torch.utils.data import Dataset
from pyannote.audio import Model, Inference

# Set the path to the Hugging Face authentication token
HUGGING_FACE_AUTH_TOKEN = ""

class SpeakerEmbeddingDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, chunk_duration=1, chunk_overlap=0, regression_model=False):
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
        self.embeddings, self.targets = self.preprocess_files()

    def preprocess_files(self):
        embeddings = []
        targets = []
        for i, file_path in enumerate(self.file_paths):
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            voiced_audio = self.apply_vad(audio)
            chunks = self.extract_chunks(voiced_audio)
            for chunk in chunks:
                tensor_chunk = torch.tensor(chunk).unsqueeze(0).float()
                embedding = self.inference({'waveform': tensor_chunk, 'sample_rate': self.sample_rate})
                embeddings.append(embedding)
                targets.append(self.labels[i])  # Use the label from the CSV
        targets = np.asarray(targets)
        if self.regression_model:
            targets = torch.from_numpy(targets).type(torch.FloatTensor)
        else:
            targets = torch.from_numpy(targets).type(torch.LongTensor)
        return embeddings, targets

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
        return torch.tensor(self.embeddings[idx]), self.targets[idx]
