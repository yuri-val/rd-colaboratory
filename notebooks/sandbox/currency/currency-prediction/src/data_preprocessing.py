import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load the CSV data into a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the data by handling missing values."""
    # Fill missing values with the mean of the column
    data.fillna(data.mean(), inplace=True)
    return data

def normalize_data(data):
    """Normalize the data using Min-Max scaling."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(file_path, target_column):
    """Load, clean, normalize, and split the data."""
    data = load_data(file_path)
    data = clean_data(data)
    data = normalize_data(data)
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    return X_train, X_test, y_train, y_test