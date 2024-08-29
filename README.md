# Car Price Prediction

This project focuses on predicting the prices of used cars based on various features using a linear regression model. The project includes data cleaning, exploratory data analysis (EDA), model training, and evaluation.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [License](#license)

## Dataset
The dataset used in this project is `quikr_car.csv`, which contains information about used cars, including:
- `name`: The name of the car.
- `company`: The manufacturer of the car.
- `year`: The year the car was manufactured.
- `Price`: The selling price of the car.
- `kms_driven`: The total kilometers driven by the car.
- `fuel_type`: The type of fuel used by the car.

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries (install via `requirements.txt` if provided):
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Instructions
1. Clone this repository.
2. Ensure that the necessary libraries are installed.
3. Place the dataset `quikr_car.csv` in the project directory.

## Data Cleaning

The dataset undergoes several cleaning steps:
- **Year**: Non-numeric values are removed, and the column is converted to integer type.
- **Price**: Rows with 'Ask For Price' are removed, and the prices are converted to integer after removing commas.
- **Kms Driven**: The 'kms' suffix is removed, and the column is converted to integer.
- **Fuel Type**: Rows with missing values are dropped.
- **Car Name**: Only the first three words of the car name are retained.

The cleaned data is then used for further analysis and model training.

## Exploratory Data Analysis (EDA)

Several visualizations are created to explore the relationship between the features and the price of the cars:
- **Company vs. Price**: Boxplot to understand the distribution of prices across different companies.
- **Year vs. Price**: Swarm plot to see how the price varies with the manufacturing year.
- **Kms Driven vs. Price**: Relationship between kilometers driven and price.
- **Fuel Type vs. Price**: Boxplot to analyze the impact of fuel type on price.
- **Combined Analysis**: Visualization combining company, fuel type, year, and price.

## Model Training

A linear regression model is trained to predict car prices based on the following features:
- `name`
- `company`
- `year`
- `kms_driven`
- `fuel_type`

### Steps:
1. **OneHotEncoding**: Categorical features (`name`, `company`, `fuel_type`) are transformed using OneHotEncoder.
2. **ColumnTransformer**: A column transformer is used to apply the OneHotEncoder to the categorical columns while passing the remaining columns unchanged.
3. **Pipeline**: A pipeline is created that first applies the column transformer and then fits the linear regression model.

## Model Evaluation

The model is evaluated using the R2 score, which measures the goodness of fit. The best model was found using different random states in the train-test split, achieving an R2 score of approximately 0.92.

## Usage

To predict the price of a car using the trained model:

```python
import pandas as pd
import numpy as np
import pickle

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Predict the price for a specific car
car_features = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=np.array(['Mahindra Jeep CL550', 'Mahindra', 2010, 1500, 'Petrol']).reshape(1, 5))
predicted_price = pipe.predict(car_features)
print(predicted_price)
