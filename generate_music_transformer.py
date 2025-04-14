import numpy as np
import tensorflow as tf
import pretty_midi
import random
import os

# Load Transformer model
model = tf.keras.models.load_model("trained_model/dual_transformer_model.keras")

# Constants
GENERATED_FOLDER = "generated_music"
os.makedirs(GENERATED_FOLDER, exist_ok=True)
GENERATE_STEPS = 240
TEMPERATURE = 1.2

print("Generating Transformer-based music for all songs...")

def temperature_sample(probabilities, temperature=1.0):
    probs = np.log(probabilities + 1e-8) / temperature
    exp_probs = np.exp(probs)
    return exp_probs / np.sum(exp_probs)

def generate_sequence(seed, steps=240, temp=1.0):
    generated_drums = []
    generated_melody = []
    input_seq = seed.copy()

    for _ in range(steps):
        input_batch = np.expand_dims(input_seq, axis=0)
        pred_drums, pred_melody = model.predict(input_batch, verbose=0)

        pred_drums = temperature_sample(pred_drums, temp)
        pred_melody = temperature_sample(pred_melody, temp)

        next_drums = (np.random.rand(*pred_drums.shape) < pred_drums).astype(np.float32).flatten()
        next_melody = (np.random.rand(*pred_melody.shape) < pred_melody).astype(np.float32).flatten()

        generated_drums.append(next_drums)
        generated_melody.append(next_melody)

        next_combined = np.concatenate([next_drums, next_melody])[np.newaxis, :]
        input_seq = np.vstack([input_seq[1:], next_combined])

    return np.array(generated_drums), np.array(generated_melody)

def save_dual_midi(drums, melody, index_to_pitch, output_file):
    midi = pretty_midi.PrettyMIDI()
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)
    melody_track = pretty_midi.Instrument(program=0, is_drum=False)
    step_time = 0.25
    DRUM_DIM = drums.shape[1]

    for t, (drum_vec, mel_vec) in enumerate(zip(drums, melody)):
        time = t * step_time
        for i, val in enumerate(drum_vec):
            if val >= 0.3:
                pitch = index_to_pitch[i]
                note = pretty_midi.Note(velocity=random.randint(90, 120), pitch=pitch,
                                        start=time, end=time + 0.22)
                drum_track.notes.append(note)
        for i, val in enumerate(mel_vec):
            idx = i + DRUM_DIM
            if val >= 0.5:
                pitch = index_to_pitch[idx]
                note = pretty_midi.Note(velocity=random.randint(70, 110), pitch=pitch,
                                        start=time, end=time + random.uniform(0.3, 0.5))
                melody_track.notes.append(note)

    midi.instruments.append(drum_track)
    midi.instruments.append(melody_track)
    midi.write(output_file)
    print(f"Saved: {output_file}")

# Loop through all preprocessed .npz files
for filename in os.listdir("processed_data"):
    if filename.endswith(".npz"):
        filepath = os.path.join("processed_data", filename)
        print(f"\nProcessing {filename}...")

        data = np.load(filepath, allow_pickle=True)
        X = data["X"]
        y_drums = data["y_drums"]
        y_melody = data["y_melody"]
        index_to_pitch = data["index_to_pitch"].tolist()

        # Generate from last seed
        seed = X[-1]
        gen_drums, gen_melody = generate_sequence(seed, steps=GENERATE_STEPS, temp=TEMPERATURE)

        # Save output
        song_name = filename.replace("_dual.npz", "").replace(".npz", "")
        output_path = os.path.join(GENERATED_FOLDER, f"{song_name}_Transformer.mid")
        save_dual_midi(gen_drums, gen_melody, index_to_pitch, output_path)
