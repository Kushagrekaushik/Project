import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense

def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X = np.load('X.npy')
    y = np.load('y.npy')
    
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=10, batch_size=32)
    model.save('eeg_model.h5')
