from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import serial
import threading
import time
import collections
from datetime import datetime

app = Flask(__name__)

# Load model and encoder
try:
    model = joblib.load("weather_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

# Data structure for IoT Monitoring
data_lock = threading.Lock()
sensor_data = {
    "temperature": 0,
    "humidity": 0,
    "precipitation": 0,
    "uv_index": 0,
    "prediction": "Calculating...",
    "status": "Connected",
    "timestamp": "",
    "confidence": "High"
}
history = collections.deque(maxlen=20)

def read_serial():
    global sensor_data
    try:
        # In a real environment, use COM7. Here we simulate or handle error.
        ser = serial.Serial("COM7", 9600, timeout=1)
        sensor_data["status"] = "Connected"
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                values = line.split(",")
                if len(values) == 4:
                    with data_lock:
                        update_values(float(values[0]), float(values[1]), float(values[2]), float(values[3]))
            time.sleep(0.5)
    except Exception as e:
        print(f"Serial error: {e}")
        sensor_data["status"] = "Simulation Mode"
        # Simulation for Premium UI testing
        while True:
            with data_lock:
                t = round(25 + np.sin(time.time()/10)*5 + np.random.normal(0, 0.5), 1)
                h = round(60 + np.cos(time.time()/12)*10 + np.random.normal(0, 1), 1)
                p = round(max(0, min(100, 20 + np.sin(time.time()/15)*40 + np.random.normal(0, 5))), 1)
                uv = round(max(0, 5 + np.sin(time.time()/20)*6 + np.random.normal(0, 0.5)), 1)
                update_values(t, h, p, uv)
            time.sleep(2)

def update_values(t, h, p, uv):
    sensor_data["temperature"] = t
    sensor_data["humidity"] = h
    sensor_data["precipitation"] = p
    sensor_data["uv_index"] = uv
    sensor_data["timestamp"] = datetime.now().strftime("%H:%M:%S")
    
    input_df = pd.DataFrame([[t, h, p, uv]], columns=["Temperature", "Humidity", "Precipitation (%)", "UV Index"])
    pred = model.predict(input_df)
    sensor_data["prediction"] = label_encoder.inverse_transform(pred)[0]
    
    history.append({
        "temperature": t, 
        "humidity": h, 
        "precipitation": p, 
        "uv_index": uv, 
        "timestamp": sensor_data["timestamp"]
    })

thread = threading.Thread(target=read_serial, daemon=True)
thread.start()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/temp_humidity')
def temp_humidity(): return render_template('temp_humidity.html')

@app.route('/precipitation')
def precipitation(): return render_template('precipitation.html')

@app.route('/uv_index')
def uv_index(): return render_template('uv_index.html')

@app.route('/prediction')
def prediction(): return render_template('prediction.html')

@app.route('/data')
def get_data():
    with data_lock:
        return jsonify({"current": sensor_data, "history": list(history)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
