import streamlit as st 
from datetime import date
import pandas as pd 
import pickle
import xgboost as xgb
import sklearn
import numpy as np 


st.title("Predict Injury Type")

# labels for inputs 
all_features = pickle.load(open("all_features.pkl", 'rb'))

all_features = pickle.load(open("common_features.pkl", 'rb'))

scaler = pickle.load(open("scaler.pkl", 'rb'))

encoded = pickle.load(open("encoder.pkl", 'rb'))

xgboost = pickle.load(open("xgboost.pkl", 'rb'))

df=pd.read_csv('train_data.csv')


categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.extend([
        column for column in df.columns if 'Code' in column or 'County' in column or 'Carrier' in column or 'Agreement'
])

numerical_features = [
        column for column in df.select_dtypes(include=['int64', 'float64']).columns
        if 'Code' not in column and 'County' not in column and 'Carrier' not in column and 'Decision' not in column and 'Indicator' not in column  and 'Grouped' not in column
    ]

categorical_features_filtered =  [item for item in categorical_features if 'Description' not in item]





app_cat_feat =list(set(categorical_features_filtered) & set(all_features))




numerical_features = [item for item in all_features if item not in app_cat_feat]
categorical= []
numerical = []
inputs = {}


uniques= {}
for cat in app_cat_feat:
    if(cat in all_features):
        categorical.append(cat)
        arr = df[cat].unique()
        arr = arr[~np.array([x is None or x != x for x in arr])]
        
        inputs[cat] = st.selectbox(cat,arr)




inputs['Average Weekly Wage'] =  st.number_input('Average Weekly Wage',min_value=0, max_value=1000000, value=0, step=1000)

binary_col =  [column for column in numerical_features if 'Binary' in column]

log_col = [column for column in numerical_features if 'log' in column.lower()]




for col in binary_col:
    sel_date = st.checkbox("Add " + col.replace("Binary","")+"?")
    if sel_date:
        col_name = col.replace("Binary","")
        # Show date input only if "No Date" is unchecked
        date_c3 = st.date_input(
            col_name,
            min_value=date(2000, 1, 1),
            max_value=date(2023, 12, 31),
            key=col_name
        )
        inputs[col] =  1
    else:
        inputs[col] =  0



date_columns = [col for col in all_features if 'date' in col.lower() or 'days' in col.lower()]

for col in date_columns:
    if("Binary" not in col):
        col = col.replace("Log","").strip()
        if('First Hearing' in col):
            inputs["First Hearing Date"] = st.date_input("First Hearing Date :")
            inputs["First Hearing Date"] =  pd.to_datetime(inputs["First Hearing Date"])
        else:
            inputs[col] = st.date_input(col+":")
            inputs[col] =  pd.to_datetime(inputs[col])



inputs['Accident Date'] = st.date_input(
    "Accident Date:",
    min_value=date(2000, 1, 1),
    max_value=date(2022, 12, 31)
)

inputs['Accident Date'] =  pd.to_datetime(inputs['Accident Date'])

intersection_keys = encoded.keys() & inputs.keys()

inputs = pd.DataFrame([inputs])


inputs['Attorney/Representative'] = inputs['Attorney/Representative'].map({'Y': 1, 'N': 0})
#categorical encoding
for key in intersection_keys:
    
    inputs[key] = encoded[key].transform(inputs[key])
    
#days calculation
filtered_keys = [key for key in inputs if 'Day' in key or 'Date' in key]
for col in filtered_keys:
    days_col = col.replace('Date', 'Days')
    if('Binary' not in col):
        
        inputs[days_col] = (inputs[col] - inputs['Accident Date']).dt.days
    else:
        inputs[days_col]= 1



inputs.drop(columns=[col for col in inputs.columns if 'date' in col.lower() and not 'binary' in col.lower()] , inplace=True )


date_columns = [col for col in inputs.columns if 'log' in col.lower()]


for col in log_col:
    col_name  = col.replace("Log","").strip()
    
    col_name2  = col.replace("log","").strip()
    
    if(col_name in inputs.keys()):
        inputs[col] = np.log(inputs[col_name]) if inputs[col_name].any() > 0 else 0
        continue 
    if(col_name2 in inputs.keys()):
        inputs[col] = np.log(inputs[col_name2]) if inputs[col_name2].any() > 0 else 0
        continue
    if(col == "First Hearing Days"):
        inputs["Log First Hearing Days"] = np.log(inputs[col_name2]) if inputs[col_name2].any() > 0 else 0

inputs.drop(columns=['C-2 Days Binary','C-3 Days Binary','First Hearing Days','Accident Days','Average Weekly Wage'],inplace=True)

features_order = pickle.load(open("features_order.pkl", 'rb'))


inputs.to_csv('inputs.csv', index=False)
inputs= inputs[all_features]



inputs = scaler.transform(inputs)




with st.form(key='my_form'):
    # Optionally, add other form elements here
    submit_button = st.form_submit_button("Submit")

    # Handle form submission
    if submit_button:
        prediction = xgboost.predict(inputs)
        st.write(f"The prediction is {prediction[0]}")