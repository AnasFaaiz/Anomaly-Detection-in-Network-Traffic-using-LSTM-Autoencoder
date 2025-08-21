
# Anomaly Detection in Network Traffic using LSTM Autoencoder

This project demonstrates an unsupervised approach to detecting anomalies in network traffic data using a Long Short-Term Memory (LSTM) Autoencoder. The model is built with TensorFlow/Keras and trained on synthetic data representing normal network behavior.

# 📜 Overview
The core idea is to train an autoencoder model to learn the patterns of normal network traffic. An autoencoder is a type of neural network that learns to compress (encode) and then reconstruct (decode) its input.

When the model is trained exclusively on normal data, it becomes very good at reconstructing normal sequences, resulting in a low reconstruction error. However, when it encounters an anomalous sequence that deviates from the learned patterns, it will struggle to reconstruct it accurately, leading to a high reconstruction error. By setting a threshold on this error, we can effectively distinguish between normal and anomalous traffic.

# 💡 How It Works
The anomaly detection process follows these key steps:
 * Data Generation: Synthetic network traffic data is generated using numpy. The dataset contains a majority of "normal" data points and a small fraction of "anomalous" points, which have a different statistical distribution.
   
 * Preprocessing:
   * The data is normalized to a [0, 1] scale using MinMaxScaler to ensure all features contribute equally to the model's training.
   * The continuous data is then transformed into overlapping sequences, which serve as the input for the LSTM model. This is crucial for capturing time-series patterns.
     
 * Model Architecture: An LSTM Autoencoder is constructed.
   * Encoder: An LSTM layer reads the input sequence and compresses it into a fixed-size vector representation, capturing the temporal dynamics.
   * Decoder: Another LSTM layer takes this vector and attempts to reconstruct the original input sequence.
   * Output: A TimeDistributed(Dense) layer ensures the output has the same shape as the input sequence.
 * Training: The model is trained only on normal sequences. This forces the autoencoder to learn the latent representation and temporal structure of what constitutes "normal" behavior.
   
 * Anomaly Detection:
   * The trained model is used to predict (reconstruct) all sequences in the dataset (both normal and anomalous).
   * The Mean Squared Error (MSE) between the original sequences and their reconstructions is calculated.
   * A threshold is determined (in this case, the 95th percentile of the reconstruction errors).
   * Any sequence with an MSE above this threshold is flagged as an anomaly.
   
📂 Project Structure
/
├── detection.py      # Main Python script with the implementation
└── README.md         # This readme file

⚙️ Prerequisites
 * Python 3.7+
 * The required libraries are listed in requirements.txt.
🚀 Installation & Usage
 * Clone the repository:
   git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

 * Install the required packages:
   It is recommended to use a virtual environment.
   pip install -r requirements.txt

   If you don't have a requirements.txt file, you can create one with the following content:
   numpy
pandas
scikit-learn
tensorflow
matplotlib

   Then, install using the command above.
 * Run the script:
   python detection.py

⚖️ License
This project is licensed under the MIT License. See the LICENSE file for details.
