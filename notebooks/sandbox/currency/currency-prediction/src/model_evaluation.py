from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2, y_pred

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='orange')
    plt.title('Currency Prediction: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.show()

def main(model_path, X_test, y_test):
    model = load_model(model_path)
    mse, mae, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')
    
    plot_predictions(y_test, y_pred)

# Example usage:
# if __name__ == "__main__":
#     model_path = '../models/trained_model.pkl'
#     X_test = pd.DataFrame(...)  # Load or prepare your test features
#     y_test = pd.Series(...)      # Load or prepare your test labels
#     main(model_path, X_test, y_test)