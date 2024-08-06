import serial
import time

def receive_data(serial_port='/dev/ttyUSB0', baud_rate=9600, output_file='eeg_data.txt'):
    ser = serial.Serial(serial_port, baud_rate)
    with open(output_file, 'w') as f:
        while True:
            line = ser.readline().decode('utf-8').strip()
            f.write(f"{line}\n")
            print(f"Received: {line}")

if __name__ == "__main__":
    receive_data()
