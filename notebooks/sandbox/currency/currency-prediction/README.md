# Currency Prediction Project

This project aims to develop a generative model for predicting currency exchange rates based on historical data. The model is trained using various features derived from the currency exchange rates, enabling it to make accurate predictions for future dates.

## Project Structure

- **data/**: Contains the dataset used for training the model.
  - **transformed_currency_data.csv**: Historical currency data with exchange rates and relevant features.

- **src/**: Source code for data processing, model training, evaluation, and prediction.
  - **data_preprocessing.py**: Handles loading, cleaning, and preparing the data for training.
  - **model_training.py**: Implements the training process for the generative model.
  - **model_evaluation.py**: Evaluates the model's performance using various metrics.
  - **prediction.py**: Makes predictions using the trained model.

- **models/**: Contains the trained model.
  - **trained_model.pkl**: Serialized version of the trained generative model.

- **notebooks/**: Jupyter notebook for exploratory data analysis.
  - **exploratory_data_analysis.ipynb**: Contains visualizations and statistical summaries of the data.

- **requirements.txt**: Lists the Python dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd currency-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data preprocessing script to prepare the data:
   ```
   python src/data_preprocessing.py
   ```

4. Train the model:
   ```
   python src/model_training.py
   ```

5. Evaluate the model:
   ```
   python src/model_evaluation.py
   ```

6. Make predictions:
   ```
   python src/prediction.py
   ```

## Usage Examples

- To train the model, run the `model_training.py` script after preprocessing the data.
- Use the `prediction.py` script to input new data and receive predicted exchange rates.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.