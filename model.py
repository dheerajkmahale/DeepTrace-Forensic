import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm_model(input_shape=(224, 224, 3)):
    """Builds a CNN-LSTM model for deepfake detection."""
    # CNN for frame feature extraction
    cnn = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])
    
    # LSTM for temporal dependency
    model = models.Sequential([
        layers.TimeDistributed(cnn, input_shape=(10, 224, 224, 3)),
        layers.LSTM(64),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# To be used in training script
if __name__ == "__main__":
    model = build_cnn_lstm_model()
    model.summary()
