import serial
import numpy as np
import tensorflow as tf
from preprocess.preprocess_data import *

# Load trained model
model = tf.keras.models.load_model('dqn_wheelchair_model.h5')

# Connect to Arduino
ser_input = serial.Serial('/dev/ttyACM0', 9600)
ser_output = serial.Serial('/dev/ttyACM1', 9600)

eeg_data = { 'C3': [], 'Cz': [], 'C4': [] }

def send_command(command):
    ser_output.write(command.encode())

while True:
    if ser_input.in_waiting > 0:
        line = ser_input.readline().decode().strip()
        signalC3, signalCz, signalC4 = map(int, line.split(","))
        eeg_data['C3'].append(signalC3)
        eeg_data['Cz'].append(signalCz)
        eeg_data['C4'].append(signalC4)

        if len(eeg_data['C3']) >= 256:
            data_array = [np.array(eeg_data['C3']), np.array(eeg_data['Cz']), np.array(eeg_data['C4'])]
            features = extract_features(data_array)
            features = np.reshape(features, [1, -1])  # Reshape for model input

            prediction = model.predict(features)
            action = np.argmax(prediction[0])

            if action == 1:
                send_command('F')  # Forward
            else:
                send_command('B')  # Backward

            eeg_data['C3'] = eeg_data['C3'][1:]  
            eeg_data['Cz'] = eeg_data['Cz'][1:]
            eeg_data['C4'] = eeg_data['C4'][1:]
