# Advanced Time Series Forecasting using Machine Learning

## Overview

This project presents an advanced approach to time series forecasting, focusing on predicting weekly and daily sales figures for retail stores. By integrating multiple data sources and employing robust preprocessing, feature engineering, and modeling techniques, the project aims to provide accurate and actionable sales predictions to support strategic decision-making in retail management.

## Features

- **Data Integration:** Combines sales, store, and holiday datasets to form a comprehensive dataset for analysis.
- **Data Preprocessing:** Handles missing values, renames columns for clarity, and ensures proper data types.
- **Feature Engineering:** Extracts temporal features, applies one-hot encoding to categorical variables, and prepares data for modeling.
- **Model Training and Evaluation:** Utilizes XGBoost for regression tasks, performs hyperparameter tuning with Optuna, and evaluates models using metrics like MSE, MAE, and R².
- **Negative Prediction Handling:** Implements safeguards to replace negative predictions with zero, ensuring validity in sales forecasting.
- **Visualization:** Generates plots to compare actual vs. predicted sales on daily and weekly scales, and conducts residual analysis to assess model performance.
- **Future Sales Forecasting:** Extends the model to predict future sales by simulating future datasets and applying the trained model.

## Data

The project utilizes the following datasets:

- **Sales Data (`sales.csv`):** Historical sales records containing information on sales amounts, store numbers, categories, and promotions.
- **Store Data (`stores.csv`):** Details about each store, including store number and type.
- **Holiday Data (`holidays_events.csv`):** Information on holidays and special events that may influence sales.
- **Future Simulation Data (`future_dataset.csv`):** Simulated data for future dates to enable forecasting beyond the historical period.

## Methodologies

### Data Loading and Integration

- **Merging Datasets:** Combines sales, store, and holiday data based on common keys (`store_nbr` and `date`) to form a unified dataset, enabling holistic analysis of factors affecting sales.

### Data Preprocessing

- **Renaming Columns:** Standardizes column names for consistency and clarity, facilitating easier data manipulation and analysis.
- **Handling Missing Values:** Addresses missing data by filling or imputing values to maintain data integrity and prevent biases in the model.
- **Date Handling:** Converts date columns to datetime objects and sets them as indices, enabling effective time series operations and feature extraction.

### Feature Engineering

- **Temporal Features:** Extracts year, month, day, day of the week, and weekend indicators from the date to capture seasonal and temporal patterns in sales data.
- **Categorical Encoding:** Applies one-hot encoding to categorical variables such as category, city, state, store type, and holiday information, allowing the model to interpret categorical data effectively.

### Model Training and Evaluation

- **Model Selection:** Chooses XGBoost as the primary model due to its superior performance in initial evaluations and its ability to handle complex relationships in the data.
- **Hyperparameter Tuning:** Utilizes GridSearchCV with TimeSeriesSplit for cross-validation to identify optimal hyperparameters, enhancing model performance.
- **Performance Metrics:** Evaluates models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score to assess accuracy and reliability.

### Prediction Refinement

- **Handling Negative Predictions:** Sets a `base_score` and enforces a lower bound to prevent the model from predicting negative sales values, ensuring realistic and meaningful forecasts.
- **Post-Processing:** Replaces any remaining negative predictions with zero, further ensuring the validity of the sales predictions.

### Visualization and Residual Analysis

- **Sales Trends:** Plots actual vs. predicted sales on daily and weekly scales to visualize model performance and identify trends.
- **Residual Analysis:** Creates scatter plots and histograms of residuals to detect any systematic errors or biases in the predictions, providing insights for further model improvement.

### Future Sales Forecasting

- **Data Simulation:** Extends the dataset with dummy categorical data for future dates to enable forecasting beyond the historical period.
- **Pipeline Application:** Applies the same preprocessing and feature engineering pipeline to the combined historical and future dataset, ensuring consistency in data preparation.
- **Model Deployment:** Trains the model on the entire historical dataset and predicts future sales based on the simulated future data, providing projections for strategic planning.

## Technologies & Libraries

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Hyperparameter Optimization:** GridSearchCV
- **Environment:** Jupyter Notebook


## Results

The XGBoost model demonstrates robust performance in forecasting sales, effectively capturing temporal dependencies and categorical influences. Visualization of actual vs. predicted sales on both daily and weekly scales shows strong alignment, indicating the model's accuracy. Residual analyses reveal minimal systematic errors, underscoring the model's reliability. Negative predictions are successfully mitigated through implemented safeguards, ensuring the practicality of the sales forecasts.

## Future Work

- **Incorporate Additional Features:** Explore more advanced feature engineering techniques, such as incorporating external economic indicators or promotional events, to enhance model performance.
- **Advanced Hyperparameter Tuning:** Utilize more sophisticated hyperparameter optimization methods like Optuna for improved efficiency and potentially better model performance.
- **Model Ensemble:** Combine multiple models to create an ensemble, which may enhance forecasting accuracy and robustness.
- **Deployment:** Develop a real-time forecasting dashboard or integrate the model into a prod
