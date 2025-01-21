# Federated Learning and Neural Networks for Climate Ads Classification  

This project demonstrates how to classify climate-related advertisements using a **neural network** for text classification. The project applies **federated learning** to train models across multiple datasets without centralizing data, ensuring data privacy while leveraging powerful machine learning techniques.

---

## Key Highlights  

### **1. Neural Network for Multi-Class Classification**  
- A **deep neural network** is built using **TensorFlow** and Keras for classifying advertisements into different typologies based on text features.
- The model includes **dense layers** with **ReLU activations**, **dropout regularization** to avoid overfitting, and a **softmax output layer** for multi-class classification.
- **TF-IDF vectorization** is used for feature extraction from text, transforming raw advertisement data into a format suitable for training.

### **2. Federated Learning for Data Privacy**  
- Implemented **Flower (FL)**, a federated learning framework, to simulate decentralized training across multiple clients (datasets).
- Clients train locally on their datasets and only share model weights, preserving data privacy.
- The global model is updated using the **FedAvg (Federated Averaging)** technique, where the weights from each client are averaged to update the global model.

### **3. Evaluation Metrics**  
- **Accuracy**, **classification reports**, and **confusion matrices** are used to assess model performance across each client and globally.

---

## Neural Network Architecture  

The neural network is built using **Keras** in TensorFlow with the following layers:

```python
model = Sequential()
model.add(Dense(512, input_dim=self.X_train.shape[1], activation='relu'))  # Dense layer
model.add(Dropout(0.2))  # Regularization
model.add(Dense(256, activation='relu'))  # Dense layer
model.add(Dropout(0.2))  # Regularization
model.add(Dense(128, activation='relu'))  # Dense layer
model.add(Dropout(0.2))  # Regularization
model.add(Dense(max_classes, activation='softmax'))  # Output layer for multi-class classification
Dense Layers: Learn high-level representations from the data.
ReLU Activation: Efficiently models non-linear relationships in the data.
Dropout: Reduces overfitting by randomly deactivating neurons during training.
Softmax Output: Converts raw scores into probabilities for multi-class classification.
ML Workflow
1. Data Preprocessing
The text data (e.g., ad_creative_body) is cleaned and vectorized using TF-IDF to convert text into numerical features suitable for neural network input:

python
Copy
Edit
def preprocess_text(text):
    text = text.lower()
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text
TF-IDF Vectorization:
python
Copy
Edit
global_vectorizer = TfidfVectorizer(stop_words='english')
global_vectorizer.fit(all_text)
2. Training the Neural Network
The neural network is trained on each client's dataset, where the model learns to predict ad typologies from the text. Training is done using the Adam optimizer with sparse categorical cross-entropy loss:

python
Copy
Edit
model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)
3. Federated Learning
Federated learning is implemented to train the neural network across multiple clients (datasets) without sharing sensitive data:

python
Copy
Edit
def federated_training(clients, num_rounds=10):
    global_weights = clients[0].model.get_weights()
    for round_num in range(num_rounds):
        local_weights = []
        for client in clients:
            client.set_parameters(global_weights)
            client.train()  # Train the model locally
            local_weights.append(client.get_parameters())
        
        # Federated Averaging (FedAvg)
        avg_weights = [np.mean(np.array([local_weights[i][j] for i in range(len(local_weights))]), axis=0) for j in range(len(local_weights[0]))]
        global_weights = avg_weights

        # Update global model
        for client in clients:
            client.set_parameters(global_weights)
Federated Averaging (FedAvg): Combines local models from each client by averaging their weights.
4. Evaluation
The model is evaluated after each round of federated learning to track performance:

python
Copy
Edit
accuracy, report, conf_matrix = client.evaluate()
Evaluation metrics include accuracy, precision, recall, F1-score, and the confusion matrix to measure how well the model is performing.
How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/m-krishanu07/federated-learning-climate-ads.git  
cd federated-learning-climate-ads  
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt  
Place your datasets:
Ensure datasets like train.csv and test.csv are in the same directory as the script.

Run the Jupyter Notebook:
Open the Climateobs.ipynb notebook in Jupyter and run each cell to execute the federated learning training and evaluation process.

Results
The neural network architecture achieved good performance on the climate advertisement classification task, with high accuracy and robust classification metrics across federated clients.
Federated learning allowed for decentralized training and ensured that sensitive data remained on local devices.
Why This Matters
Machine Learning Expertise:

Implemented a deep neural network with ReLU activations and dropout to improve model accuracy and generalization.
Used TF-IDF for feature extraction and prepared data for training.
Evaluated the model rigorously with advanced metrics such as F1-score and confusion matrices.
Federated Learning:

Demonstrated how federated learning can enable distributed training across multiple clients while maintaining data privacy.
Practical Application:

Applied these machine learning techniques to a real-world task—classifying climate-related ads—showing the project’s relevance to the growing field of data privacy and sustainability.
Future Work
Model Improvements: Add advanced techniques such as LSTMs or Transformers for better handling of sequential data.
Hyperparameter Tuning: Optimize hyperparameters to improve model performance.
Scalability: Expand the federated learning setup to support a larger number of clients and datasets.
Contact
For any questions or feedback, feel free to reach out via GitHub.
