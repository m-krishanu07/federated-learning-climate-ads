
Federated Learning with Climate Data
This project demonstrates how to apply federated learning using climate-related datasets. It uses Python, TensorFlow, and Flower (FL) to build a collaborative model while keeping the data decentralized.

Key Features
Federated Learning Framework: Built with Flower (FL) for decentralized machine learning.
TF-IDF Vectorization: Extracts meaningful features from text data (like climate observations).
Neural Network: A multi-layered model designed for multi-class classification tasks.
Evaluation Metrics: Tracks accuracy, classification reports, and confusion matrices for performance evaluation.
Dataset Overview
The project works with CSV datasets, such as:

test.csv, train.csv, all.csv, etc.
Each dataset contains columns like ad_creative_body (text) and Typology 1 (labels). Missing columns are flagged during preprocessing.
How It Works
Preprocessing:

Cleans text by removing special characters and converting to lowercase.
Uses TF-IDF to transform text into numeric features.
Federated Clients:
Each client trains a local model on its dataset and shares weights for global aggregation.

Federated Averaging (FedAvg):
Combines weights from all clients into a global model.

Evaluation:
Clients are evaluated on their test data after each federated round.

Code Highlights
Text Preprocessing:

python
Copy
Edit
def preprocess_text(text):
    return ''.join(e for e in text.lower() if e.isalnum() or e.isspace())
Cleans text for TF-IDF vectorization.

TF-IDF Vectorization:

python
Copy
Edit
global_vectorizer = TfidfVectorizer(stop_words='english')
global_vectorizer.fit(all_text)
Converts text into features for machine learning.

Federated Client Training:

python
Copy
Edit
client.train()
Trains each client’s local neural network model.

Federated Aggregation:

python
Copy
Edit
avg_weights = [np.mean(np.array([local_weights[i][j] for i in range(len(local_weights))]), axis=0) for j in range(len(local_weights[0]))]
Combines local models into a global model using the FedAvg algorithm.

How to Run
Install Requirements:

Copy
Edit
pip install -r requirements.txt
Ensure Flower, TensorFlow, and required libraries are installed.

Prepare Data:
Place your CSV files in the same directory as the script. Ensure the column names match (ad_creative_body, Typology 1).

Run the Script:

Copy
Edit
python Climateobs.ipynb
Monitor Training:
Track federated rounds and client evaluations directly in the output logs.

Why It Matters
Federated Learning: Ensures privacy by keeping raw data local.
Text Preprocessing: Handles messy text data effectively.
TF-IDF + Neural Networks: Combines traditional text analysis with deep learning for accurate predictions.
