import pickle
import pandas as pd
import streamlit as st


global lang_model

lang = open('LRmodelCAR.pkl','rb')
lang_model=pickle.load(lang)
lang.close()
# Giving title to webpage:
st.title("Car Price Predictor :")
# Adding text box for inputing text form the user:
input_test = st.text_input("Enter Car's Name👇","Maruti Suzuki Swift")
# Adding text box for inputing text form the user:
input_test2 = st.text_input("Enter Companys's Name👇","Maruti")
# Adding text box for inputing text form the user:
input_test3 = st.number_input("Enter Year of Manufacturing 👇")
# Adding text box for inputing text form the user:
input_test4 = st.number_input("Enter Kilometer driven 👇")
# Adding text box for inputing text form the user:
input_test5 = st.text_input("Enter fuel type(Petrol/Diesel)👇","Petrol")

# now making a submit button:
button_submit = st.button("Get Price")
# making above button work:
if button_submit:
    output = lang_model.predict(pd.DataFrame([[input_test ,input_test2,input_test3,input_test4,input_test5]],columns=['name','company','year','kms_driven','fuel_type']))
    end = output[0]
    st.text(end)