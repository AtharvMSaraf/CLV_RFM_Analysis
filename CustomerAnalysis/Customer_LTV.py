import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import math

# import plydata.cat_tools as cat
import plotnine as pn

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV

pn.options.dpi = 300

def read_data(path, initial_analysis):
    df = pd.read_csv(path,index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    if initial_analysis:
        print(df.head(5))
        print(df.info())
        print(df.isnull().sum())
        print(df.describe())
    return df
def cohort_Analysis(df):
    cohort_first_purchase_tbl = df.sort_values(['customer_id','date']).groupby('customer_id').first()
    # print(cohort_first_purchase_tbl)
    print('First date of purchase within cohort: ',cohort_first_purchase_tbl['date'].min())
    print('Last date of purchase within cohort: ',cohort_first_purchase_tbl['date'].max())

    # Monthly total 
    # df['Month'] = df['date'].dt.to_period('M')
    # df.groupby('Month')["price"].sum().plot()
    # plt.ylabel('SUM')
    # plt.show()

    # Visualise: Inidvidual Customer Purchases
    ids = df['customer_id'].unique()
    print('Total unique id = ', len(ids))
    selected_ids = ids[:10]
    cust_subset_df = df[df['customer_id'].isin(selected_ids)]
    
    num_plots = len(selected_ids)
    fig, ax = plt.subplots(4, 3, figsize=(10,10))
    ax = ax.flatten()
    for i, customer_id in enumerate(selected_ids):
        group = cust_subset_df[cust_subset_df['customer_id'] == customer_id]
        ax[i].plot(group['date'], group['price'], marker='o', label=f'Customer ID {customer_id}')
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel('Value')
        ax[i].set_title(f'Customer ID {customer_id}')
        ax[i].tick_params(axis='x', rotation=90)
    
    # Delete unused subplots
    for i in range(num_plots, 4*3):
            fig.delaxes(ax[i])

    plt.tight_layout()
    plt.show()


# Time splitting
def feature_Engineering(df):
     
    # Temporal splitting
    number_Of_Days_Threshold = 90
    max_date = df['date'].max()
    split_date = max_date - pd.to_timedelta(number_Of_Days_Threshold,unit='d')
    
    df_before_threshold = df[df['date'] <= split_date ]        # From this data we will extract our features  
    df_after_threshold = df[df['date'] > split_date ]          # This data will be used as targets

    # Taget generation 
    target_df = df_after_threshold.drop(['quantity','date'], axis = 1)\
                .groupby("customer_id")\
                .sum()\
                .rename({'price':'spend_90_total'}, axis = 1)\
                .assign(spend_90_flag = 1)

    # RFM Feature generation

    # 1] Recency [date] :
    #   difference in days from split date to customers most recent purchase
    recency_feature_df = df_before_threshold.drop('quantity', axis = 1)\
                        .groupby('customer_id')\
                        .apply(
                            lambda x: (x['date'].max() - split_date) / pd.to_timedelta(1, unit='d')
                        )\
                        .to_frame()\
                        .set_axis(["recency"], axis = 1)
    
    # 2] Frequency Feature [Count]:
    #       we add the qunity for each customer to generate 
    frequency_feature_df = df_before_threshold[['customer_id', 'date']]\
                            .groupby('customer_id')\
                            .count()\
                            .rename({'date': 'frequency'},axis =1)

    # 3] Monetory Features:
    #

    monetory_feature_df = df_before_threshold.groupby('customer_id')\
                            .aggregate(
                                 {
                                      'price': ["sum", "mean"]
                                 }
                            )\
                            .set_axis(['price sum', 'price_mean'], axis =1)        
    
    # Merging the features and target
    features_df = pd.concat([recency_feature_df,frequency_feature_df,monetory_feature_df], axis = 1)\
    .merge(target_df, left_index=True,right_index=True,how="left")\
    .fillna(0)
    return features_df

def model_prediction(feature_df):
    # This function predicts:
    #    -  how much each indivisdual will spen in 90 days 
    #    - probablity of spending in next 90 days

    X = feature_df[['recency','frequency', 'price sum', 'price_mean']]  # -----------  Features stay common to both model -------

    # ----------------------------------------------------------------------------------------------------------------------

    # prediction spend in next 90 days
    
    y_90_spend = feature_df['spend_90_total']

    xgb_reg_spec = XGBRegressor(objective= "reg:squarederror", random_state = 123)

    xgb_reg_model = GridSearchCV(estimator=xgb_reg_spec, param_grid= dict(learning_rate = [0.05, 0.01,0.1,0.3,0.5]), scoring= "neg_mean_absolute_error", refit = True, cv = 5)

    xgb_reg_model.fit(X,y_90_spend)

    print('Best score = ', xgb_reg_model.best_score_)
    print('Best parameters = ', xgb_reg_model.best_params_)

    amt_spend_predictions = xgb_reg_model.predict(X)

    # -------------------------------------------------------------------------------------------------------------------------------

    # predicting the probablity of spend in next 90 days
    y_spend_prob = feature_df['spend_90_flag']

    xgb_classifier_spec =  XGBClassifier(objective = "binary:logistic", random_state = 123)

    xgb_classifier_model = GridSearchCV( estimator=xgb_classifier_spec, param_grid= dict(learning_rate = [0.05,0.01,0.1,0.3]), scoring='roc_auc', refit=True,cv = 5)

    xgb_classifier_model.fit(X,y_spend_prob)

    print('Best score = ', xgb_classifier_model.best_score_)
    print('Best parameters = ', xgb_classifier_model.best_params_)

    prob_spend_prediction = xgb_classifier_model.predict_proba(X)



    # ----------------------------------------------------------------------------------------------------------------------------------

    # Feature Importance 

    # -  For amt spend in 90 days model 

    imp_feature_amt_spend = xgb_reg_model.best_estimator_.get_booster().get_score(importance_type = 'gain')
    imp_feature_prob_spend = xgb_classifier_model.best_estimator_.get_booster().get_score(importance_type = 'gain')


    fig, ax = plt.subplots(1,2,figsize = (20,10))
    fig.suptitle("Feature Importance")

    ax[0].bar(imp_feature_amt_spend.keys(), imp_feature_amt_spend.values())
    ax[0].set_title("For amount spend in 90 days ")
    ax[0].set_xlabel('Features')
    ax[0].set_ylabel('Value')

    # - for probablity of spend in 90 days  ----

    ax[1].bar(imp_feature_prob_spend.keys(), imp_feature_prob_spend.values())
    ax[1].set_title("For probablity of spend in 90 days")
    ax[1].set_xlabel("Features")
    ax[1].set_ylabel('Value')
    # plt.show()

    # ---------------- Saving the Data -----------------------------------------------
    

    # --- saving the model --- 
    joblib.dump(xgb_reg_model,'saved_data/xgb_reg_model.pkl')
    joblib.dump(xgb_classifier_model,'saved_data/xgb_classifier_model.pkl')

    # --- saving the importance ---
    joblib.dump(imp_feature_amt_spend,'saved_data/imp_feature_amt_spend.pkl')
    joblib.dump(imp_feature_prob_spend,'saved_data/imp_feature_prob_spend.pkl')

    # --- saving predicted probablity of spending in 90 days and predicted amount spent with all the other info ---
    final_df = feature_df.copy()
    final_df['pred_90_spend'] = amt_spend_predictions
    prob_spend_prediction_df = pd.DataFrame(prob_spend_prediction)
    prob_spend_prediction_df.index = feature_df.index               # both had different indexes
    final_df['pre_prob'] = prob_spend_prediction_df.iloc[:, 1]
    final_df.to_csv('saved_data/final_df.csv')


if __name__ == "__main__":
    df = read_data('/Users/atharvsaraf/Documents/DSprojects/Customer Lifetime Value & RFM analysis/CustomerAnalysis/cdnow.csv',initial_analysis=0)
    # cohort_Analysis(df)
    feature_df = feature_Engineering(df)
    predictions = model_prediction(feature_df)