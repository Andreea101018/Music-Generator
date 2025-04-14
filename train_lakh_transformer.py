import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Load all .npz files
X_list, y_drums_list, y_melody_list = [], [], []
for file in os.listdir("processed_data"):
    if file.endswith(".npz"):
        data = np.load(os.path.join("processed_data", file), allow_pickle=True)
        X_list.append(data["X"])
        y_drums_list.append(data["y_drums"])
        y_melody_list.append(data["y_melody"])

# Find minimum input dim for consistency
min_dim = min(x.shape[2] for x in X_list)
X_list = [x for x in X_list if x.shape[2] == min_dim]
y_drums_list = [y for x, y in zip(X_list, y_drums_list) if x.shape[2] == min_dim]
y_melody_list = [y for x, y in zip(X_list, y_melody_list) if x.shape[2] == min_dim]

# Concatenate all datasets
X = np.concatenate(X_list, axis=0)
y_drums = np.concatenate(y_drums_list, axis=0)
y_melody = np.concatenate(y_melody_list, axis=0)

SEQ_LEN = X.shape[1]
INPUT_DIM = X.shape[2]
DRUM_DIM = y_drums.shape[1]
MELODY_DIM = y_melody.shape[1]

print(f"Combined dataset: X={X.shape}, y_drums={y_drums.shape}, y_melody={y_melody.shape}")

# Transformer block
def transformer_block(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(rate)(ffn)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn)

# Build model
def build_model():
    inputs = Input(shape=(SEQ_LEN, INPUT_DIM))
    x = transformer_block(inputs, embed_dim=INPUT_DIM, num_heads=4, ff_dim=128)
    x = transformer_block(x, embed_dim=INPUT_DIM, num_heads=4, ff_dim=128)
    x = x[:, -1]  # Last time step

    drums_out = Dense(DRUM_DIM, activation="sigmoid", name="drums_out")(x)
    melody_out = Dense(MELODY_DIM, activation="sigmoid", name="melody_out")(x)

    model = Model(inputs=inputs, outputs=[drums_out, melody_out])
    return model

model = build_model()
model.compile(
    loss={"drums_out": "binary_crossentropy", "melody_out": "binary_crossentropy"},
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    metrics={"drums_out": "accuracy", "melody_out": "accuracy"}
)

# Callbacks
os.makedirs("trained_model", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

checkpoint = ModelCheckpoint("checkpoints/transformer_model_best.keras", monitor="loss", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X,
    {"drums_out": y_drums, "melody_out": y_melody},
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop]
)

# Save model
model.save("trained_model/dual_transformer_model.keras")
print("Transformer model trained and saved.")
