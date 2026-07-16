import tensorflow as tf
import numpy as np
from model import build_cnn_lstm_model

def train():
    # Mock data generation for demonstration as actual dataset download/prep is manual
    X_train = np.random.rand(10, 10, 224, 224, 3)
    y_train = np.random.randint(0, 2, 10)
    
    model = build_cnn_lstm_model()
    model.fit(X_train, y_train, epochs=1)
    
    # Save model
    model.save('deepfake_model.h5')
    print("Model trained and saved as deepfake_model.h5")

if __name__ == "__main__":
    train()
