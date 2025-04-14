import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Load and merge all processed data files
X_list, y_drums_list, y_melody_list = [], [], []

for filename in os.listdir("processed_data"):
    if filename.endswith("_dual.npz"):
        print(f"Loading {filename}")
        data = np.load(os.path.join("processed_data", filename), allow_pickle=True)
        X_list.append(data["X"])
        y_drums_list.append(data["y_drums"])
        y_melody_list.append(data["y_melody"])

X = np.concatenate(X_list, axis=0)
y_drums = np.concatenate(y_drums_list, axis=0)
y_melody = np.concatenate(y_melody_list, axis=0)

SEQ_LEN = X.shape[1]
INPUT_DIM = X.shape[2]
DRUM_DIM = y_drums.shape[1]
MELODY_DIM = y_melody.shape[1]

print(f"Combined data: X shape = {X.shape}, y_drums = {y_drums.shape}, y_melody = {y_melody.shape}")

# Build dual-output model
inputs = Input(shape=(SEQ_LEN, INPUT_DIM))

x = LSTM(256, return_sequences=True)(inputs)
x = Dropout(0.3)(x)
x = LSTM(256)(x)
x = Dropout(0.3)(x)

drum_output = Dense(DRUM_DIM, activation="sigmoid", name="drums")(x)
melody_output = Dense(MELODY_DIM, activation="sigmoid", name="melody")(x)

model = Model(inputs=inputs, outputs=[drum_output, melody_output])

model.compile(
    loss={"drums": "binary_crossentropy", "melody": "binary_crossentropy"},
    optimizer="adam",
    metrics={"drums": "accuracy", "melody": "accuracy"}
)

model.summary()

# Create output folders
os.makedirs("trained_model", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Save best model
checkpoint = ModelCheckpoint(
    "checkpoints/dual_model_best.keras",
    monitor="loss",
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X,
    {"drums": y_drums, "melody": y_melody},
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint]
)

# Save full model
model.save("trained_model/dual_rnn_model.keras")
print("Model trained and saved to 'trained_model/dual_rnn_model.keras'")