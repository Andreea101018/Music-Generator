import pretty_midi
import numpy as np
import os

# Config
MIDI_FOLDER = "midi_input_music"
OUTPUT_FOLDER = "processed_data"
SEQ_LENGTH = 32
TIME_RESOLUTION = 4  # steps per second

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Build global pitch vocab
global_pitch_set = set()

print("Scanning all MIDI files for pitch vocab...")
for filename in os.listdir(MIDI_FOLDER):
    if filename.endswith(".mid") or filename.endswith(".midi"):
        path = os.path.join(MIDI_FOLDER, filename)
        try:
            midi = pretty_midi.PrettyMIDI(path)
            for inst in midi.instruments:
                for note in inst.notes:
                    global_pitch_set.add(note.pitch)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# Build pitch mappings
index_to_pitch = sorted(global_pitch_set)
pitch_to_index = {p: i for i, p in enumerate(index_to_pitch)}
TOTAL_DIM = len(index_to_pitch)

print(f"Found {TOTAL_DIM} unique pitches")

# Preprocess each file with unified vocab
for filename in os.listdir(MIDI_FOLDER):
    if not (filename.endswith(".mid") or filename.endswith(".midi")):
        continue

    path = os.path.join(MIDI_FOLDER, filename)
    name = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{name}_dual.npz")

    print(f"\n Processing {filename}...")
    try:
        midi = pretty_midi.PrettyMIDI(path)
        total_time = midi.get_end_time()
        time_steps = int(total_time * TIME_RESOLUTION)

        drums = [inst for inst in midi.instruments if inst.is_drum]
        melody = [inst for inst in midi.instruments if not inst.is_drum]

        # Create multi-hot matrix
        multi_hot = np.zeros((time_steps, TOTAL_DIM), dtype=np.float32)

        for inst in drums + melody:
            for note in inst.notes:
                t = int(note.start * TIME_RESOLUTION)
                if t >= time_steps:
                    continue  # Avoid out-of-bounds
                idx = pitch_to_index.get(note.pitch)
                if idx is not None:
                    multi_hot[t, idx] = 1.0

        # Create sequences 
        X, y_drums, y_melody = [], [], []
        for i in range(len(multi_hot) - SEQ_LENGTH):
            seq = multi_hot[i:i + SEQ_LENGTH]
            target = multi_hot[i + SEQ_LENGTH]
            X.append(seq)
            y_drums.append(target[:TOTAL_DIM // 2])
            y_melody.append(target[TOTAL_DIM // 2:])

        X = np.array(X)
        y_drums = np.array(y_drums)
        y_melody = np.array(y_melody)

        np.savez(
            output_path,
            X=X,
            y_drums=y_drums,
            y_melody=y_melody,
            pitch_to_index=pitch_to_index,
            index_to_pitch=index_to_pitch
        )
        print(f"Saved to: {output_path} ({X.shape[0]} sequences)")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
