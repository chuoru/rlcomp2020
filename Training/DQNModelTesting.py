from keras.models import Sequential
from keras.layers import Dense
from keras import activations
import keras
import numpy as np
import pandas as pd

model = Sequential([Dense(32, activation = 'relu', input_shape = (784,)), Dense(10,activation = 'softmax')])
foo = np.array([12,-3,1.5,-5,0.4,-4.3], dtype = 'float64')
print(keras.activations.softmax(foo).numpy())