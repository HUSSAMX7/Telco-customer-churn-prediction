# Telco-customer-churn-prediction
A classification machine learning problem for predicting customers churn from the company based on customers who left within the last month labeled by 'yes' or 'no'

The dataset used in this project is obtained from [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)\
The data set includes information about:
- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## Data cleaning
* Convert 'TotalCharges' column which is of object type to float type using pd.to_numeric() with errors parameter set to 'coerce' to parse invalid data to NaN.
* Data has no duplicates.
  
* The TotalCharges column was originally of type object, and was converted to float using pd.to_numeric(errors='coerce') to handle invalid entries.

* Missing values in TotalCharges (found using a mask on null values) were dropped to clean the dataset.

## Feature encoding 

To prepare the data for machine learning models, a custom encoding function encode_dataframe(df) was created and applied to the dataset.

The function performs the following steps:

Label Encoding for binary categorical columns (i.e., columns with only two unique values like "Yes"/"No" or "Male"/"Female").

One-Hot Encoding for multi-class categorical features (with more than two unique values).

Boolean conversion: All boolean values resulting from encoding were converted into integer format using a lambda function to ensure model compatibility.


## Feature scaling

MinMaxScaler from sklearn.preprocessing was used to apply min-max scaling on the numerical columns: tenure, MonthlyCharges, and TotalCharges.
This transformation scales the values to a range between 0 and 1.

## Data imbalance
Data imbalance affects machine learning models by tending only to predict the majority class and ignoting the minority class, hence, having major misclassification of the minority class in comparison with the majority class. Hence, we use techniques to balance class distribution in the data.

Even that our data here doesn't have severe class imbalance, but handling it shows results improvement.
Using SMOTE (Synthetic Minority Oversampling Technique) libraray in python that randomly increasing the minority class which is 'yes' in our case.

