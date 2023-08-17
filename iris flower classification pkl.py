# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:36:18 2023

@author: tarun
"""

import pandas as pd
import numpy as np
import pickle


pickled_model = pickle.load(open("C:/Users/tarun/Downloads/iris flower classification/iris flower.pkl","rb"))

input_data = (6.2,3.4,5.4,2.3)
input_array = np.asarray(input_data)
input_reshaped_data = input_array.reshape(1,-1)

output = pickled_model.predict(input_reshaped_data)

if output == 0:
    print("Iris-setosa")
elif output == 1:
    print("Iris-versicolor")
else:
    print("Iris-virginica")