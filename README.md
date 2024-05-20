# Customer Lifetime Value & RFM Analysis

## Project Overview
This project aims to predict customer spending behavior over a 90-day period using historical transactional data. By leveraging advanced data analysis and machine learning techniques, it performs Customer Lifetime Value (CLV) analysis and RFM (Recency, Frequency, Monetary) segmentation. The main objective is to predict both the probability of a customer making a purchase and the amount they are likely to spend in the next 90 days.

## Technologies and Libraries Used
- **Pandas** and **NumPy**: Data manipulation and analysis
- **Matplotlib** and **Plotnine**: Data visualization
- **XGBoost**: Building predictive models
- **Scikit-learn**: Model selection and hyperparameter tuning
- **Joblib**: Model persistence

## Project Structure
```
.
├── data
│   └── cdnow.csv                     # Input dataset
├── saved_data
│   ├── xgb_reg_model.pkl             # Saved XGBoost regression model
│   ├── xgb_classifier_model.pkl      # Saved XGBoost classification model
│   ├── imp_feature_amt_spend.pkl     # Feature importance for regression model
│   ├── imp_feature_prob_spend.pkl    # Feature importance for classification model
│   └── final_df.csv                  # Final DataFrame with predictions
├── src
│   ├── analysis.py                   # Functions for cohort and feature analysis
│   ├── model.py                      # Model training and prediction functions
│   ├── utils.py                      # Utility functions (e.g., data loading)
│   └── main.py                       # Main script to run the analysis and modeling
├── README.md                         # Project documentation
└── requirements.txt                  # Python package dependencies
```

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
    git clone https://github.com/yourusername/customer-lifetime-value.git
    cd customer-lifetime-value
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the main script:**
    ```bash
    python src/main.py
    ```

## Future Enhancements
- **Enhance Feature Set**: Incorporate additional features such as customer demographics or product categories.
- **Advanced Models**: Explore other machine learning models or ensemble methods to improve prediction accuracy.
- **Real-time Predictions**: Implement real-time predictions and updates.
- **Deployment**: Develop a web application or API for easy access to the prediction models and insights.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or further information, please contact [your-email@example.com](mailto:your-email@example.com).
