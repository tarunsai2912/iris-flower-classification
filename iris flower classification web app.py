# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:43:17 2023

@author: tarun
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

pickled_model = pickle.load(open("C:/Users/tarun/Downloads/iris flower classification/iris flower.pkl","rb"))

def iris_flower(input_data):
    input_array = np.asarray(input_data)
    input_reshaped_data = input_array.reshape(1,-1)

    output = pickled_model.predict(input_reshaped_data)

    if output == 0:
        return("It is an Iris-setosa flower")
    elif output == 1:
        return("It is an Iris-versicolor flower")
    else:
        return("It is an Iris-virginica flower")
    
def main():
    st.title("IRIS FLOWER SPECIES CLASSIFICATION")
    st.write("Please Enter The Following Data")
    
    sepal_length = st.text_input("Enter the sepal length")
    sepal_width = st.text_input("Enter the sepal width")
    petal_length = st.text_input("Enter the petal length")
    petal_width = st.text_input("Enter the petal width")
    
    ans = ''
    
    if st.button("Classify the Flower"):
        ans = iris_flower([sepal_length,sepal_width,petal_length,petal_width])
        
    st.success(ans)
    
if __name__ == "__main__":
    main()
        