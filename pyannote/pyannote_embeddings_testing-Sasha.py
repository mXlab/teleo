import tkinter as tk
from tkinter import messagebox
import numpy as np
import sounddevice as sd
import tempfile
import wave
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from pyannote.audio import Model, Inference
import os
import glob

# Load model
HUGGING_FACE_AUTH_TOKEN = ""
model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGING_FACE_AUTH_TOKEN)
inference = Inference(model, window="whole")

# Global variables
embeddings = []
SIMILARITY_THRESHOLD = 0.5  # Define a threshold for similarity (adjust as needed)

def process_existing_files(directory="./Voices"):
    """Process all existing audio files in the specified directory."""
    global embeddings
    filepaths = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if not filepaths:
        print("No existing audio files found.")
        return

    for filepath in filepaths:
        print(f"Processing {filepath}...")
        try:
            emb = inference(filepath)  # Generate embedding
            emb = np.asarray(emb)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)  # Reshape if necessary
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if embeddings:
        update_graph()

def record_audio(duration=5, samplerate=44100): # Set the duration of the recording
    """Record audio for a given duration and save it to a temporary file."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16') # Record audio @ 16-bit for 5 seconds @ 44100 Hz
    sd.wait()  # Wait until recording is finished
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Recording saved to {temp_file.name}")
    return temp_file.name

def process_audio():
    """Record and process audio, then update the graph."""
    global embeddings # Use the global embeddings array
    try:
        audio_file = record_audio() # Record audio function
        emb = inference(audio_file)  # Generate embedding for the recorded audio
        emb = np.asarray(emb) # Convert to numpy array
        if emb.ndim == 1: # Check if the embedding is 1D
            emb = emb.reshape(1, -1)  # Reshape if necessary

        # If there are existing embeddings, check similarity
        if embeddings: # Check if there are existing embeddings
            stacked_embeddings = np.vstack(embeddings) # Stack existing embeddings
            similarities = 1 - cdist(stacked_embeddings, emb, metric='cosine').flatten() # Compute cosine similarity
            max_similarity = np.max(similarities) # Get the maximum similarity

            print(f"Maximum similarity with existing embeddings: {max_similarity}")

            # If the similarity is below the threshold, ignore the recording
            if max_similarity < SIMILARITY_THRESHOLD:
                print("The new recording is too far from existing embeddings. Ignoring.")
                messagebox.showinfo("No Match", "The new recording is too far from existing embeddings. Ignored.")
                return

        # Add the new embedding to the list
        embeddings.append(emb)

        # Update the graph
        update_graph()

    except Exception as e:
        messagebox.showerror("Error", f"Error processing audio: {e}")


def update_graph(): # Update the graph with the current embeddings
    if not embeddings:
        print("No embeddings to visualize.")
        return

    # Stack all embeddings into a matrix
    stacked_embeddings = np.vstack(embeddings)

    # Compute distances
    inv_dist = 1 - cdist(stacked_embeddings, stacked_embeddings, metric='cosine')
    print(f"Updated inverse distances: {inv_dist}")

    # Update visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(inv_dist, cmap="viridis", annot=False, cbar=True)
    plt.title("Inverse Cosine Distance between Embeddings")

    # Add a big "X" for the most similar match
    if len(embeddings) > 1:
        new_index = len(embeddings) - 1  # Index of the new recording
        # Find the most similar existing embedding
        similarities = inv_dist[new_index, :-1]  # Exclude the new recording itself
        most_similar_index = np.argmax(similarities)  # Index of the most similar embedding

        # Get the matching file name
        filepaths = sorted(glob.glob(os.path.join("./Voices", "*.wav")))
        if filepaths:
            matching_file = filepaths[most_similar_index]
            print(f"The new recording matches with: {matching_file}")
            messagebox.showinfo("Match Found", f"The new recording matches with: {matching_file}")

        # Place an "X" at the coordinates of the most similar match
        plt.text(
            most_similar_index + 0.5, new_index + 0.5, "X", color="red", fontsize=20,
            ha="center", va="center", fontweight="bold"
        )
        plt.text(
            new_index + 0.5, most_similar_index + 0.5, "X", color="red", fontsize=20,
            ha="center", va="center", fontweight="bold"
        )

    plt.show(block=False)  # Non-blocking graph display

def start_ui():
    """Start the tkinter UI."""
    root = tk.Tk()
    root.title("Audio Embedding Processor")

    # Instructions
    tk.Label(root, text="Click 'Record & Process' to record audio and update the graph.").pack(pady=10)

    # Record button
    record_button = tk.Button(root, text="Record & Process", command=process_audio, bg="green", fg="white")
    record_button.pack(pady=10)

    # Exit button
    exit_button = tk.Button(root, text="Exit", command=root.quit, bg="red", fg="white")
    exit_button.pack(pady=10)

    root.mainloop()

# Process existing files and create the base graph
process_existing_files()

# Start the UI
start_ui()


