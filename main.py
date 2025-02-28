import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from augumentation import add_noise, shift, stretch
from train import trainModel
from validate import validateModel
from evaluate import evalModel
from ftExtraction import InstantiateAttributes
from vis import visualisation

def main():
    # Directory containing folders with .wav and .txt files for each patient
    data_dir = 'data/'  # Update this path
    
    # Load and preprocess data from the directory
    X, y = InstantiateAttributes(data_dir)
    
    # Split data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply data augmentation (optional)
    # You can apply augmentation functions like add_noise, shift, stretch on your data here
    
    # Train the model
    x_train, x_test, y_train,y_test, history=trainModel(X, y)
    print("x_test=",x_test.shape)
    # Validate the model
    y_pred = validateModel(x_test)

    print(f"y_pred shape: {y_pred.shape}, y_test shape: {y_test.shape}")

    # Evaluate the model performance
    metrics = evalModel(y_test, y_pred)
    print(metrics)

    print("Visualisation graphs:")
    visualisation(y_pred, y_test, x_test, history)
if __name__ == '__main__':
    main()
