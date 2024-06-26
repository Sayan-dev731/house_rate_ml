# House Price Prediction using Linear Regression

This project uses linear regression to predict house prices based on house sizes. The dataset used is assumed to be in a CSV file named `home_dataset.csv`.

## Project Overview

- **Data Loading**: Load the dataset from a CSV file.
- **Data Visualization**: Plot the house prices against house sizes to visualize the data.
- **Data Splitting**: Split the data into training and testing sets.
- **Model Training**: Train a linear regression model on the training set.
- **Model Prediction**: Predict house prices for the test set.
- **Result Visualization**: Visualize the actual and predicted house prices.

## Requirements

- Python 3.x
- NumPy
- pandas
- matplotlib
- scikit-learn

## Setup

1. Clone the repository or download the project files.
2. Ensure you have Python 3.x installed.
3. Install the required Python libraries using pip:

    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

4. Place the `home_dataset.csv` file in the same directory as the script.

## Usage

1. Run the Python script:

    ```bash
    python house_price_prediction.py
    ```

2. The script will:
   - Load the dataset.
   - Visualize the house prices against house sizes.
   - Split the data into training and testing sets.
   - Train a linear regression model.
   - Predict house prices for the test set.
   - Visualize the actual and predicted house prices.

## Code Explanation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data from CSV file
data = pd.read_csv('home_dataset.csv')

# Extract features and target variable
house_sizes = data['HouseSize'].values
house_prices = data['HousePrice'].values

# Visualize the data
plt.scatter(house_sizes, house_prices, marker='o', color='blue')
plt.title('House Prices vs. House Size')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)

# Reshape the data for training
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict prices for the test set
predictions = model.predict(x_test)

# Print shapes to debug
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("predictions shape:", predictions.shape)

# Visualize the predictions
plt.scatter(x_test, y_test, marker='o', color='blue', label='Actual Prices')
plt.plot(x_test, predictions, color='red', linewidth=2, label='Predicted Prices')
plt.title('House Price Prediction with Linear Regression')
plt.xlabel('House Size (sq.ft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()
