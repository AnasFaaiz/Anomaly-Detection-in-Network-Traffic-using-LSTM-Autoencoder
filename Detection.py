import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import tensorflow as tf
import random

# Set reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Generate synthetic network traffic data
def generate_synthetic_data(normal_size=10000, anomaly_size=500, features=5):
    normal_data = np.random.normal(loc=0.0, scale=1.0, size=(normal_size, features))
    anomalies = np.random.normal(loc=5.0, scale=1.0, size=(anomaly_size, features))
    data = np.concatenate([normal_data, anomalies], axis=0)
    np.random.shuffle(data)
    columns = ["duration", "packet_size", "src_port", "dst_port", "protocol"]
    return pd.DataFrame(data, columns=columns)

# Normalize the data
def normalize_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

# Create sequences for LSTM input
def create_sequences(data, seq_length=10):
    return np.array([data[i:i + seq_length] for i in range(len(data) - seq_length)])

# Build LSTM Autoencoder
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)
    repeated = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Plot reconstruction error
def plot_error(errors, threshold):
    plt.figure(figsize=(10, 4))
    plt.hist(errors, bins=50, alpha=0.7, label='Reconstruction Error')
    plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error Histogram")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# MAIN
df = generate_synthetic_data()
scaled_data, scaler = normalize_data(df)
sequence_length = 10
sequences = create_sequences(scaled_data, sequence_length)

# Train only on normal sequences
normal_sequences = sequences[:10000 - sequence_length]
X_train, X_test = train_test_split(normal_sequences, test_size=0.2, random_state=SEED)

model = build_autoencoder((sequence_length, scaled_data.shape[1]))

# Train the model
history = model.fit(
    X_train, X_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)

# Predict on full data
predicted = model.predict(sequences)
mse = np.mean(np.power(sequences - predicted, 2), axis=(1, 2))

# Thresholding
threshold = np.percentile(mse, 95)
anomaly_labels = (mse > threshold).astype(int)

# Results
results_df = pd.DataFrame({
    "reconstruction_error": mse,
    "anomaly": anomaly_labels
})

# Plot error
plot_error(mse, threshold)

# Print anomaly stats
print("Anomaly Threshold:", threshold)
print(results_df["anomaly"].value_counts())
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# 1. Load data from API
def load_api_data():
    url = "https://jsonplaceholder.typicode.com/posts"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

# 2. Convert categorical/length-based features into numeric
def preprocess_api_data(df):
    df['title_length'] = df['title'].apply(len)
    df['body_length'] = df['body'].apply(len)
    df['userId'] = df['userId'].astype(float)
    df['id'] = df['id'].astype(float)
    df = df[['userId', 'id', 'title_length', 'body_length']]
    return df

# 3. Normalize and sequence
def normalize_and_sequence(df, seq_len=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    sequences = np.array([scaled[i:i + seq_len] for i in range(len(scaled) - seq_len)])
    return sequences, scaler

# 4. Build LSTM Autoencoder
def build_lstm_autoencoder(input_shape):
    inp = Input(shape=input_shape)
    encoded = LSTM(32, activation='relu', return_sequences=False)(inp)
    repeated = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(repeated)
    out = TimeDistributed(Dense(input_shape[1]))(decoded)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# 5. Visualize reconstruction error
def plot_reconstruction_error(errors, threshold):
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.show()

# Main Pipeline
df_raw = load_api_data()
df_processed = preprocess_api_data(df_raw)
sequence_length = 5
sequences, scaler = normalize_and_sequence(df_processed, sequence_length)

# Train-test split
X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

model = build_lstm_autoencoder((sequence_length, sequences.shape[2]))
model.fit(X_train, X_train, epochs=15, batch_size=32, validation_split=0.1)

# Reconstruction
X_pred = model.predict(sequences)
mse = np.mean(np.square(sequences - X_pred), axis=(1, 2))

# Threshold (95th percentile)
threshold = np.percentile(mse, 95)
anomalies = (mse > threshold).astype(int)

# Results
results = pd.DataFrame({
    "reconstruction_error": mse,
    "anomaly": anomalies
})

plot_reconstruction_error(mse, threshold)
print("Anomaly threshold:", threshold)
print(results["anomaly"].value_counts())
