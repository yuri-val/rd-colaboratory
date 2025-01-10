import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import joblib


# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(method="ffill", inplace=True)  # Forward fill to handle missing values
    return data


def preprocess_data(data):
    # Drop the 'exchangedate' column
    data = pd.DataFrame(data).drop(columns=["exchangedate"])

    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step), :])
        y.append(data[i + time_step, 0])  # Assuming the first column is the target
    return np.array(X), np.array(y)


# Define the model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Main function to train the model
def train_model():
    data = load_data("../data/transformed_currency_data.csv")
    scaled_data, scaler = preprocess_data(data)

    # Create dataset
    time_step = 10  # Number of previous time steps to consider
    X, y = create_dataset(scaled_data, time_step)

    # Split the data into training and testing sets
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor="loss", patience=5)
    model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

    # Save the model and scaler
    model.save("../models/trained_model.h5")
    joblib.dump(scaler, "../models/scaler.pkl")


if __name__ == "__main__":
    train_model()
