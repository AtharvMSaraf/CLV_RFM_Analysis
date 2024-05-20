# Customer Lifetime Value & RFM Analysis

## Project Overview
This project aims to predict customer spending behavior over a 90-day period using historical transactional data. By leveraging advanced data analysis and machine learning techniques, it performs Customer Lifetime Value (CLV) analysis and RFM (Recency, Frequency, Monetary) segmentation. The main objective is to predict both the probability of a customer making a purchase and the amount they are likely to spend in the next 90 days.

## Technologies and Libraries Used
- **Pandas** and **NumPy**: Data manipulation and analysis
- **Matplotlib** and **Plotnine**: Data visualization
- **XGBoost**: Building predictive models
- **Scikit-learn**: Model selection and hyperparameter tuning
- **Joblib**: Model persistence



## Key Components and Functionality

### Data Reading and Initial Analysis
The `read_data` function loads the transactional data from a CSV file, converts date columns to datetime format, and optionally provides an initial analysis, including summary statistics and missing values.

### Cohort Analysis
The `cohort_Analysis` function analyzes customer cohorts based on their first purchase date and visualizes individual customer purchase histories.

### Feature Engineering
The `feature_Engineering` function splits the data into training and target sets based on a specified threshold of days. It generates RFM features and merges them with the target variable to create a comprehensive feature set for modeling.

### Model Prediction
The `model_prediction` function uses XGBoost to predict the amount customers will spend in the next 90 days and the probability of them spending in the next 90 days. It uses GridSearchCV for hyperparameter tuning and visualizes feature importance for both models. The trained models, feature importance metrics, and prediction results are saved for future use.

## Visualizations and Outputs
- **Customer Purchase Histories**: Plots of individual customer purchase patterns over time.
- **Feature Importance**: Bar charts illustrating the importance of different features in the predictive models.
- **Predicted Values**: A CSV file containing the predicted spending amounts and probabilities for each customer.

## How to Run the Project

1. **Clone the repository:**
    ```bash
    git clone https://github.com/AtharvMSaraf/CLV_RFM_Analysis
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script:**
    ```bash
    python CustomerAnalyis/Customer_LTV.py
    ```


