# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16, 2023

Designed By: Ayodele Ayodeji
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16, 2023

Designed By: Ayodele Ayodeji
"""
# Importing necessary libraries
import numpy as np
import pickle
import pandas as pd
from catboost import CatBoostRegressor
import streamlit as st 

from PIL import Image

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_california_house_price(house_age, total_rooms, total_bedrooms, population, households, median_income):
  
    """
    **housingMedianAge:** Median age of a house within a block; a lower number is a newer building
    **totalRooms:** Total number of rooms within a block
    **totalBedrooms:** Total number of bedrooms within a block
    **population:** Total number of people residing within a block
    **households:** Total number of households, a group of people residing within a home unit, for a block

    """

    prediction = classifier.predict([[house_age, total_rooms, total_bedrooms, population, households, median_income]])


    print(prediction)
    return prediction



def main():
    st.title("California House Price Prediction App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit House Prices Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    house_age = st.text_input("House Age","Type Here")
    total_rooms = st.text_input("Total Rooms","Type Here")
    total_bedrooms = st.text_input("Total Bedrooms","Type Here")
    population = st.text_input("Population","Type Here")
    households = st.text_input("Households","Type Here")
    median_income = st.text_input("Median Income","Type Here")
    
    result=""
    if st.button("Predict"):
        result = predict_california_house_price(house_age, total_rooms, total_bedrooms, population, households, median_income)
        formatted_result = "${:.4f}".format(result.item())
        st.success('The Prediction Price Based on your Requirement is {}'.format(formatted_result))


    if st.button("About"):
        st.text(" Predictive Analysis Project Using Python")
        st.text(" Background of the dataset")
        st.text("""
            The data contains information from the 1990 California census. 
            So although it may not help you with predicting current housing prices like the Zillow Zestimate dataset,
            it does provide an accessible introductory dataset for teaching people about the basics of machine learning.
            The data pertains to the houses found in a given California district and some summary stats 
            about them based on the 1990 census data. Be warned the data aren't cleaned 
            so there are some preprocessing steps required! The columns are as follows, 
            their names are pretty self explanitory:longitude, latitude, housing_median_age, 
            total_rooms, total_bedrooms, population, households, median_income, median_house_value, 
            ocean_proximity.

            variable descrition:

            longitude: A measure of how far west a house is; a higher value is farther west.

            latitude: A measure of how far north a house is; a higher value is farther north.

            housingMedianAge: Median age of a house within a block; a lower number is a newer building.

            totalRooms: Total number of rooms within a block.

            totalBedrooms: Total number of bedrooms within a block.

            population: Total number of people residing within a block.

            households: Total number of households, a group of people residing within a home unit, for a block.

            medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars).

            medianHouseValue: Median house value for households within a block (measured in US Dollars).

            oceanProximity: Location of the house w.r.t ocean/sea.
            
                """)

              
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
  
