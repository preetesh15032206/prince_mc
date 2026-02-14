# Weather Prediction ML Project

## Overview
A machine learning project that uses a Random Forest Classifier to predict weather types (e.g., Sunny, Rainy) based on sensor values like temperature, humidity, precipitation, and UV index.

## Project Architecture
- **train_model.py** - Trains the weather classification model using the CSV dataset and saves the model/encoder as `.pkl` files
- **predict_weather.py** - Makes weather predictions using hardcoded sensor values (main entry point)
- **live_predict.py** - Reads live sensor data from an Arduino via serial port (requires hardware)
- **weather_classification_data.csv** - Training dataset
- **weather_model.pkl** - Pre-trained Random Forest model
- **label_encoder.pkl** - Label encoder for weather type categories

## How to Run
- Run `predict_weather.py` to make a prediction with example sensor values
- Modify the temperature, humidity, precipitation, and UV index values in `predict_weather.py` to test different scenarios
- Run `train_model.py` to retrain the model from the CSV data

## Dependencies
- Python 3.12
- pandas
- scikit-learn
- joblib

## Recent Changes
- 2026-02-14: Initial setup in Replit environment
