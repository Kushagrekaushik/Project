# import serial
# import numpy as np
# import tensorflow as tf
# from preprocess_data import *

# # Load trained model
# # model = tf.keras.models.load_model('Project\predict\dqn_wheelchair_model.h5')
# # print("Model loaded successfully.")

# try:
#     model = tf.keras.models.load_model('Project\\predict\\dqn_wheelchair_model.keras')
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model: {e}")

# # Connect to Arduino
# print('KK')
# ser_input = serial.Serial('COM3', 230400)
# # ser_output = serial.Serial('COM3', 9600)

# eeg_data = {'C3': [], 'Cz': [], 'C4': []}

# # Main loop
# while True:
#     if ser_input.in_waiting > 0:
#         line = ser_input.readline().strip()

#         # Decode the line if it's a bytes object
#         if isinstance(line, bytes):
#             try:
#                 line = line.decode('utf-8', errors='ignore')
#             except UnicodeDecodeError:
#                 print(f"Could not decode line: {line}")
#                 continue

#         try:
#             # Split the line into signals and convert them to integers
#             signalC3, signalCz, signalC4 = map(float, line.split())
#             eeg_data['C3'].append(signalC3)
#             eeg_data['Cz'].append(signalCz)
#             eeg_data['C4'].append(signalC4)

#             if len(eeg_data['Cz']) >= 250:
#                 data_array = [np.array(eeg_data['Cz'])]
#                 features = extract_features(data_array)
#                 features = np.reshape(features, [1, -1])  # Reshape for model input

#                 prediction = model.predict(features)
#                 action = np.argmax(prediction[0])

#                 if action == 1:
#                     print('F')  # Forward
#                 else:
#                     print('B')  # Backward

#                 # Trim the data to keep the buffer size constant
#                 eeg_data['C3'] = eeg_data['C3'][1:]
#                 eeg_data['Cz'] = eeg_data['Cz'][1:]
#                 eeg_data['C4'] = eeg_data['C4'][1:]

#         except ValueError:
#             print(f"Error processing line: {line}")


import serial
import numpy as np
import tensorflow as tf
from preprocess_data import *
import logging

tf.get_logger().setLevel(logging.DEBUG)
# Attempt to load the trained model
try:
    model = tf.keras.models.load_model(r'Project\predict\dqn_wheelchair_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Optionally, exit the program if the model cannot be loaded
    exit()

# Connect to Arduino
print('KK')
ser_input = serial.Serial('COM3', 230400)
# ser_output = serial.Serial('COM3', 9600)

eeg_data = {'C3': [], 'Cz': [], 'C4': []}

# Main loop
while True:
    if ser_input.in_waiting > 0:
        line = ser_input.readline().strip()

        # Decode the line if it's a bytes object
        if isinstance(line, bytes):
            try:
                line = line.decode('utf-8', errors='ignore')
            except UnicodeDecodeError:
                print(f"Could not decode line: {line}")
                continue

        try:
            signalC3, signalCz, signalC4 = map(float, line.split())
            eeg_data['C3'].append(signalC3)
            eeg_data['Cz'].append(signalCz)
            eeg_data['C4'].append(signalC4)

            if len(eeg_data['Cz']) >= 250:
                data_array = [np.array(eeg_data['Cz'])]
                features = extract_features(data_array)
                features = np.reshape(features, [1, -1])  

                prediction = model.predict(features)
                action = np.argmax(prediction[0])

                if action == 1:
                    print('F')  # Forward
                else:
                    print('B')  # Backward

            
                eeg_data['C3'] = eeg_data['C3'][1:]
                eeg_data['Cz'] = eeg_data['Cz'][1:]
                eeg_data['C4'] = eeg_data['C4'][1:]

        except ValueError:
            print(f"Error processing line: {line}")
