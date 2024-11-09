# End To End Project
## Customer Churn Prediction using ANN
This project focuses on predicting customer churn using a Neural Network model (ANN). The goal is to identify customers who are likely to exit (churn) based on various features. The model is trained on a dataset with key customer information, including demographics and account details.

### Use model

link : https://customerchurnprediction-avlmup8prqzroeqnzwbiug.streamlit.app/

![alt text](<Screenshot (67).png>) 
![alt text](<Screenshot (68).png>)

### Table of Contents
Project Overview
Dataset
Technologies and Libraries Used
Data Preprocessing
Encoding Categorical Features
Splitting Data
Feature Scaling
Model Development
Architecture
Training
Evaluation
Saving Encoders and Scaler
Deployment with Streamlit

### Project Overview
Customer churn prediction is crucial for businesses to retain customers and improve satisfaction. In this project, an Artificial Neural Network (ANN) model is used to predict customer churn based on features such as credit score, age, balance, and whether the customer has a credit card or is an active member.

### Dataset
The dataset contains the following columns:

- CreditScore: Customer's credit score
- Gender: Customer's gender (Male or Female)
- Age: Customer's age
- Tenure: Number of years the customer has been with the bank
- Balance: Account balance
- NumOfProducts: Number of bank products the customer has
- HasCrCard: Whether the customer has a credit card (1 = Yes, 0 = No)
- IsActiveMember: Whether the customer is an active member (1 = Yes, 0 = No)
- EstimatedSalary: Estimated salary of the customer
- Exited: Target variable (1 = Customer churned, 0 = Customer retained)
The dataset's target column, Exited, indicates whether the customer churned.

### Technologies and Libraries Used
Python for model development and deployment
TensorFlow & Keras for building the ANN model
Pandas & Numpy for data manipulation and analysis
Scikit-Learn for preprocessing and evaluation
Pickle to save and load the encoder and scaler objects
Streamlit for deploying the web application
Data Preprocessing
Encoding Categorical Features
Geography: One-hot encoding is applied to this column to convert it into multiple binary columns.
Gender: Label encoding is used to convert gender into numerical form (0 for Female, 1 for Male).
Splitting Data
The dataset is split into training and testing sets using train_test_split from Scikit-Learn.

### Feature Scaling
A StandardScaler is applied to standardize features, which is essential for neural networks to perform optimally.

### Model Development
Architecture
The ANN model is built using Keras with a simple feedforward architecture:

Input layer matching the number of input features
Two hidden layers with ReLU activation
Output layer with sigmoid activation for binary classification

### Training
The model is trained with the following configuration:

Optimizer: Adam
Loss: Binary Crossentropy (suitable for binary classification tasks)
Metrics: Accuracy

### Evaluation
The model is evaluated on the test set to gauge its performance in predicting customer churn.

### Saving Encoders and Scaler
The LabelEncoder for gender, OneHotEncoder for geography, and StandardScaler for features are saved using the pickle library.
These saved objects are loaded during deployment to ensure consistency in preprocessing.

### Deployment with Streamlit
The model is deployed using Streamlit to create an interactive web application where users can input customer details to predict churn probability.
