import pickle
import pandas as pd
import streamlit as st


global lang_model

lang = open('flight_rf.pkl','rb')
lang_model=pickle.load(lang)
lang.close()

# Giving title to webpage:
st.title("Flight Fare predictor :")
# Adding text box for inputing text form the user:
input_test = st.text_input("Enter Total Number of stops👇")

# now making a submit button:
button_submit = st.button("Get Price")
# making above button work:
# if button_submit:
#     output = lang_model.predict(pd.DataFrame([[input_test ,input_test2,input_test3,input_test4,input_test5]],columns=['name','company','year','kms_driven','fuel_type']))
#     end = output[0]
#     st.text(end)