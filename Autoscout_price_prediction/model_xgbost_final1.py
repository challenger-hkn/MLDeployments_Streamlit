# import datetime
# from dis import show_code
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import streamlit as st
# from numpy.random import sample
from PIL import Image
pd.set_option('display.float_format', lambda x: '%.3f' % x)
st.set_page_config(page_title="The Car's Valuation", page_icon="🐞", layout="centered")

html_temp = """
<div style="background-color:Lavender;padding:4px">
<h1 style="color:MidnightBlue;text-align:center; font-size: 44px;">The Car's Value Prediction</h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: red;font-size: 44px;'>The Car's Value Prediction</h1>", unsafe_allow_html=True)
st.write("""
This app predicts the **Car Price**!
""")# st.title("The Car's Valuation") üstteki ile aynı title = # 

new_title = '<p style="font-family:sans-serif; color:Green; fontStyle:bold; font-size: 36px;">What is your car worth?</p>'
st.markdown(new_title, unsafe_allow_html=True)


def user_input_features():
    make = st.selectbox("Select Make: " , ["Audi", "Opel", "Renault"] )
    model= st.selectbox("Select Model:", ["A1", "A3", "Astra", "Clio", "Corsa", "Duster", "Espace", "Insignia"] )
    # last_year= pd.datetime.now().year # 1923 ile last year arasında 
    Year =st.number_input("Select Year", 1923,2019,2018)# 2020 sen ne seçersen ekranda ilk o olacak
    age= 2019 - Year
    Gearing_Type= st.selectbox("Select Gearing Type:" , ["Automatic", "Manual","Semi-automatic" ] )
    km=st.slider("Select km", 0 , 350000, 40000)
    # st.write("Your km is", km)
    data = {"make": make, "model": model,'age': age,'km': km, "Gearing_Type": Gearing_Type }
    features = pd.DataFrame(data, index=[0])
    return features

new_df= user_input_features()

st.write(" **Your selection** ")
st.write(new_df)

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("final_scout_short_notdummy_listed.csv")
new_df = pd.DataFrame(new_df, index=[0])

X= df.drop(columns='price')
use_df = pd.concat([new_df, X], axis=0)

cat = X.select_dtypes("object").columns
enc = OrdinalEncoder()
use_df[cat] = enc.fit_transform(use_df[cat])
new_df = use_df[:1]

loaded_model= pickle.load(open("XGBoost_model.pkl", 'rb'))
prediction= loaded_model.predict(new_df)

result = int(round(prediction[0],0))
st.write(f" The model success is **% {result:.2f}** ")

prediction=int(prediction)
if st.button("Predict"):
    st.title(f" Your car price is :   € {prediction}")
    st.success(f" **Please contact with customer services representative, mail: autoscout@gmail.com**")

img = Image.open("car.jpg")
st.image(img,caption="",width=700)

st.write("# general price")
st.line_chart(df.price)
