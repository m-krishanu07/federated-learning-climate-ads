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
