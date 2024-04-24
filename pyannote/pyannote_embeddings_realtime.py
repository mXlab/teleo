import sounddevice as sd
import numpy as np
import torch  # Import torch
from scipy.signal import resample
from pyannote.audio import Model, Inference
from queue import Queue
import time
import webrtcvad

HUGGING_FACE_AUTH_TOKEN=""

# Type of model.
regression_model = False

# Classification/regression model filename.
model_filename = 'trustnet.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
device

custom_model = torch.load(model_filename, map_location=device)


n_speakers = 4

# Parameters
sample_rate = 16000  # Audio sample rate
duration = 10  # Duration of audio chunks in seconds
frame_duration = 0.02  # Frame duration in seconds (20 ms)
frame_samples = int(sample_rate * frame_duration)  # Number of samples per frame
channels = 1  # Mono audio

# VAD setup
vad = webrtcvad.Vad()
# Set aggressiveness from 0 to 3 (3 is the most aggressive)
vad.set_mode(3)

# Instantiate pretrained model
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGING_FACE_AUTH_TOKEN)
inference = Inference(model, window="whole")

# Queue to hold audio data
audio_queue = Queue()

# Buffer for accumulating audio data
audio_buffer = np.array([])

def audio_callback(indata, frames, time, status):
    """This is called for each audio chunk from the microphone."""
    global audio_buffer
    if status:
        print(status)

    if frames == frame_samples:  # Ensure frame size is as expected
        audio_frame = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(audio_frame, sample_rate)
        if is_speech:
            # Accumulate audio data in the buffer
            audio_buffer = np.append(audio_buffer, indata[:, 0])

            # When buffer reaches 5 seconds of audio, process it
            if len(audio_buffer) >= sample_rate * duration:
                # Optionally resample here if needed
                audio_queue.put(audio_buffer[:sample_rate * duration])
                # Remove processed data from buffer
                audio_buffer = audio_buffer[sample_rate * duration:]
                print("Append to audio queue")

# Start recording from the microphone
with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback, blocksize=frame_samples):
    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            # Check if there is new data in the queue
            if not audio_queue.empty():
                data = audio_queue.get()
                # Convert numpy array to PyTorch tensor and add channel dimension
                tensor_data = torch.tensor(data).unsqueeze(0).float()
                # Ensure it has shape (channel, time)
                embedding = inference({'waveform': tensor_data, 'sample_rate': sample_rate})
                
                # Assuming `embedding` is your numpy array input
                # Convert numpy array to tensor
                embedding_tensor = torch.from_numpy(embedding).float()  # Ensure dtype is float for neural network processing

                # Move tensor to the correct device
                embedding_tensor = embedding_tensor.to(device)

                # Now you can pass this tensor to your model
                custom_model.eval()  # Ensure the model is in evaluation mode
                predicted_output = custom_model(embedding_tensor)
                print(predicted_output.shape)

                # Optional: Convert to probabilities and get class indices
                predicted_class = torch.argmax(predicted_output, dim=0)

                print("Computed embedding:", embedding.shape)
                print("Computed class:", predicted_output, predicted_class)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped recording.")
