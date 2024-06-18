import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model for Car price predictor 
lang = open('LRmodelCAR.pkl','rb')
lang_model=pickle.load(lang)
lang.close()

# Load the trained model for airplane fare price 
model_path = 'flight_rf.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Set up seaborn style
sns.set()
def main():

    st.sidebar.header('User Input Features')
    def user_input_features():
        Date_of_Journey = st.sidebar.date_input('Date of Journey', datetime.today())
        Journey_day = Date_of_Journey.day
        Journey_month = Date_of_Journey.month

        Dep_Time = st.sidebar.time_input('Departure Time', datetime.now())
        Dep_hour = Dep_Time.hour
        Dep_min = Dep_Time.minute

        Arrival_Time = st.sidebar.time_input('Arrival Time', datetime.now())
        Arrival_hour = Arrival_Time.hour
        Arrival_min = Arrival_Time.minute

        Duration_hours = st.sidebar.number_input('Duration Hours', min_value=0, max_value=24, value=1)
        Duration_mins = st.sidebar.number_input('Duration Minutes', min_value=0, max_value=59, value=0)

        Total_Stops = st.sidebar.selectbox('Total Stops', [0, 1, 2, 3, 4])

        Airline = st.sidebar.selectbox('Airline', ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 
                                                   'Vistara', 'GoAir', 'Multiple carriers', 
                                                   'Multiple carriers Premium economy', 
                                                   'Jet Airways Business', 'Vistara Premium economy', 'Trujet'])
        
        Source = st.sidebar.selectbox('Source', ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore'])
        
        Destination = st.sidebar.selectbox('Destination', ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata', 'Bangalore'])

        data = {'Total_Stops': Total_Stops,
                'Journey_day': Journey_day,
                'Journey_month': Journey_month,
                'Dep_hour': Dep_hour,
                'Dep_min': Dep_min,
                'Arrival_hour': Arrival_hour,
                'Arrival_min': Arrival_min,
                'Duration_hours': Duration_hours,
                'Duration_mins': Duration_mins,
                'Airline': Airline,
                'Source': Source,
                'Destination': Destination}
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Preprocess user input
    st.write("User Input Features")
    st.write(input_df)

    # Convert categorical data into numerical data using one hot encoding
    airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara', 'GoAir', 
                'Multiple carriers', 'Multiple carriers Premium economy', 
                'Jet Airways Business', 'Vistara Premium economy', 'Trujet']
    
    sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore']
    
    destinations = ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata', 'Bangalore']

    for airline in airlines:
        input_df[airline] = 0
    input_df[input_df['Airline'][0]] = 1

    for source in sources:
        input_df[source] = 0
    input_df[input_df['Source'][0]] = 1

    for destination in destinations:
        input_df[destination] = 0
    input_df[input_df['Destination'][0]] = 1

    input_df.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)

    # Ensure the input_df has the necessary columns
    required_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=required_columns, fill_value=0)

    # Make predictions
    if st.button("Predict"):
        prediction = model.predict(input_df)
        st.write(f"Predicted Flight Price: ₹{prediction[0]:.2f}")


# Giving title to webpage:
st.title("Price Predictor :")

# creating a drop box / select box

option = st.selectbox("What would you like to predict 👇",("Select","Car Price 🏎️","Airplane fare price 🛫"))

if(option == "Select"):
    st.write("Please select an option from the dropdown menu.")
elif(option == "Car Price 🏎️"):
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
else:
    if __name__ == '__main__':
        main()