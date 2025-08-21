# Anomaly Detection in Network Traffic using LSTM Autoencoder

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project provides an unsupervised learning solution for detecting anomalies in time-series data, specifically demonstrated on synthetic network traffic. It leverages a Long Short-Term Memory (LSTM) Autoencoder built with TensorFlow/Keras to identify unusual patterns that deviate from normal behavior.

## 📜 Core Concept

The fundamental principle is to train an autoencoder model to "master" the patterns of **normal** data. An autoencoder learns to compress its input into a compact representation (encoding) and then reconstruct it back to its original form (decoding).

-   **For Normal Data:** When trained exclusively on normal sequences, the model becomes highly proficient at this reconstruction task, resulting in a very low **reconstruction error** (e.g., Mean Squared Error).
-   **For Anomalous Data:** When the model encounters an anomaly—a pattern it has never seen during training—it struggles to reconstruct it accurately. This failure leads to a significantly **high reconstruction error**.

By setting a threshold on this error, we can effectively flag data points as either normal or anomalous.

## 🔧 How It Works

1.  **Data Generation**: The script generates synthetic network traffic, creating a dataset composed mostly of normal data points with a small, distinct subset of anomalies.
2.  **Preprocessing**: Data is normalized using `MinMaxScaler` and then transformed into overlapping time-series sequences suitable for an LSTM model.
3.  **Model Architecture**: A simple LSTM Autoencoder is constructed with an encoder to learn data representation and a decoder to reconstruct the input sequence.
4.  **Training**: The model is trained **only on normal data sequences**, teaching it the definition of "normal."
5.  **Inference & Detection**: The trained model predicts on the entire dataset. The reconstruction error (MSE) is calculated for each sequence, and a threshold (the 95th percentile of errors) is used to classify sequences as anomalies.

## 📂 Project Structure

.
├── detection.py
├── requirements.txt
└── README.md


## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

# 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

On macOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate

On Windows:
Bash

python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.
Bash
pip install -r requirements.txt

▶️ How to Run

Once the setup is complete, execute the main script from your terminal:
Bash
python detection.py
