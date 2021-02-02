import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
colsample_bytree= 0.8166911250194073
gamma=1.0880194042053906
max_depth= 10
reg_lambda= 497.47428517801467
subsample= 0.9424502628308582
xgb=XGBClassifier(max_depth =max_depth,subsample=subsample,colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,tree_method='gpu_hist',gamma=gamma,predictor='gpu_predictor',
                  n_estimators=1000,learning_rate=0.05)
st.title('home-site-quote-conversion-prediction')
st.header('Now please follow the steps and enter the details.')
def get_requried_columns(path):
    with open(path,'rb') as f:
        file=pickle.load(f)
    return file
def get_one_hot_encoding(path):
    with open(path,'rb') as f:
        file=pickle.load(f)
    return file
def get_scaler(path):
    with open(path,'rb') as f:
        file=pickle.load(f)
    return file

def get_weights(path):
    with open(path,'rb') as f:
        file=pickle.load(f)
    return file
def get_value(value,dtype):
    if dtype.startswith('int'):
        return int(value)
    else:
        return float(value)
def get_month(month):
    month_cos=np.cos(2*np.pi*(int(month)-1)/11)
    month_sin=np.sin(2*np.pi*(int(month)-1)/11)
    return month_sin,month_cos
def get_weekday(week):
    week_cos=np.cos(2*np.pi*(week-1)/6)
    week_sin=np.sin(2*np.pi*(week-1)/6)
    return week_sin,week_cos
numeric_columns=get_requried_columns('numeric_columns.pkl')
numeric_columns=numeric_columns+['month','weekday']
encoding_dict=get_one_hot_encoding('categorical_encoding.pkl')
dtypes=pd.read_csv('dtypes.csv')
scaler=get_scaler('standard_scaler.pkl')
#weights=get_weights()
weights=None
category_columns=list(encoding_dict.keys())
data={}
dtypes.head()
for c in numeric_columns:
    if (c!='month') and (c!='weekday'):
        value=st.text_input(f"enter the value for {c}",'1')
        if value:
            dtype=str(dtypes[dtypes['columns']==c]['dtype'])
            data[c]=get_value(value,dtype)
    elif c=='month':
        value=st.selectbox('select the requried month',[1,2,3,4,5,6,7,8,9,10,11,12])
        if value:
            data['month_sin'],data['month_cos']=get_month(int(value))
    elif c=='weekday':
        value=st.selectbox('select the requried weekday',[1,2,3,4,5,6,7])
        if value:
            data['week_sin'],data['week_cos']=get_month(int(value))
data=pd.DataFrame(data,index=[0])
for col in category_columns:
    categories=encoding_dict.get(col).get('categories')
    transformer=encoding_dict.get(col).get('transformer')
    value=st.selectbox(f"select the right category",categories)
    if value:
        one_hotted=transformer.transform([[value]])
        data=data.join(pd.DataFrame(one_hotted,columns=categories))    
predict=st.button('submit')
model_path='xgboost_model'
if predict:
    model_data=data.values
    model_data=scaler.transform(model_data)
    xgb.load_model(model_path)
    predictions=xgb.predict(model_data)
    st.success(f"the prediction is {predictions[0]}")
    

    