TabNet Stroke Prediction Project
Overview
This project uses TabNet neural network to predict stroke occurrence in patients based on health and demographic factors.
Objective
Build an accurate machine learning model to predict stroke risk and identify key contributing factors for preventive healthcare decisions.
Data Source
Stroke Prediction Dataset:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
Key Features
Data Processing

Handle missing values in BMI data
Convert categorical variables to numerical format (gender, work type, smoking status, etc.)
Normalize all features for optimal model performance

Model Training

TabNet classifier with optimized parameters
80-20 train-test split with stratified sampling
Early stopping to prevent overfitting
Adam optimizer with learning rate scheduling

Evaluation

Classification accuracy on training and validation sets
ROC curve analysis with AUC calculation
Feature importance ranking
Comprehensive performance metrics

Outputs

Trained model saved as tabnet_model.zip
Performance report with detailed metrics
Visualizations including ROC curve and feature importance plots
Training progress charts

Technical Stack

PyTorch TabNet for deep learning
Pandas and NumPy for data processing
Scikit-learn for evaluation metrics
Matplotlib and Seaborn for visualization

Results
The model provides high accuracy stroke prediction with interpretable feature importance, helping healthcare professionals make informed preventive decisions.
