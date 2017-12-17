# Based on https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://medium.com/@pushkarmandot/build-your-first-deep-learning-neural-network-model-using-keras-in-python-a90b5864116d

# Expecting CSV data in format:
# Country (string), avg, max, min, refugees, timestep (0 to 443)

import csv

import numpy
from numpy import array
from numpy import asarray
from numpy import argmax
from numpy import arange
from numpy import unique
from numpy import reshape

import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

number_of_values_per_country = 444

train_amount = 424
test_amount = 0

train_range = range(0, train_amount)
test_range = range(train_amount, train_amount + test_amount)

# Using data from previous months
lookback = 0

def load_data(file):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    countries_train = []
    countries_test = []

    with open(file, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rows:
            country, avg, max, min, refugees, time = row
            if (int(time) in train_range):
                x_train.append((int(time), float(avg), float(max), float(min), int(refugees)))
                y_train.append((int(time), int(refugees)))
                countries_train.append(country)
            if (int(time) in test_range):
                x_test.append((int(time), float(avg), float(max), float(min), int(refugees)))
                y_test.append((int(time), int(refugees)))
                countries_test.append(country)

    countries_train = array(countries_train)
    countries_test = array(countries_test)

    # Hot one encoding of countries
    b1 ,countries_encoded_train = unique(countries_train, return_inverse=True)
    b2 ,countries_encoded_test = unique(countries_test, return_inverse=True)

    x_train = zip (countries_encoded_train, x_train)
    y_train = zip (countries_encoded_train, y_train)

    # format now (country, time, avg, max, min, refugees)
    # convert to [country] + [avg,max,min] + lookback * [avg,max,min,refugees]

    last_values = []
    x_train_final = []
    y_train_final = []

    for (country, (time, avg, max, min, refugees)) in x_train:
        if (time == 0):
            del last_values[:]
        if (lookback == 0):
            x_train_final.append([country,avg,max,min])
        elif (time >= lookback):
            x_train_final.append([country,avg,max,min] + last_values)
            last_values.pop(0)
            last_values.pop(0)
            last_values.pop(0)
            last_values.pop(0)
            last_values.append(avg)
            last_values.append(max)
            last_values.append(min)
            last_values.append(refugees)
        else:
            last_values.append(avg)
            last_values.append(max)
            last_values.append(min)
            last_values.append(refugees)    

    # Update  y_test and y_train accordingly

    for (country, (time, refugees)) in y_train:
        if (time >= lookback):
            y_train_final.append(refugees)

    return (x_train_final, y_train_final)

# fix random seed for reproducibility
#numpy.random.seed(7)

# load the dataset

lookback = 12

(x_all, y_all) = load_data("LESS_DATA_CORRECTED_17_12.csv")
#x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = 0.1)

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(0,len(list(x_all))):
    if ((i % 444) < 420):
        x_train.append(x_all[i])
        y_train.append(y_all[i])
    else:
        x_test.append(x_all[i])
        y_test.append(y_all[i])

x_train = asarray(x_train)
y_train = asarray(y_train).reshape(-1,1)
x_test = asarray(x_test)
y_test = asarray(y_test).reshape(-1,1)

# scale values
scX = StandardScaler()
scY = StandardScaler()
x_train_scaled = scX.fit_transform(x_train)
x_test_scaled = scX.transform(x_test)

y_train_scaled = scY.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scY.transform(y_test.reshape(-1,1))

input_dimension = 4 + lookback * 4

x_train = x_train_scaled
x_test = x_test_scaled
y_train = y_train_scaled
y_test = y_test_scaled

# create model 1
model = Sequential()
model.add(Dense(units =  4 + lookback * 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dimension))
model.add(Dense(units = 4 + lookback * 2, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

# Or load model
model = load_model('./models/saved_model.h5')

# TRAIN
history = model.fit(x_train, y_train, epochs=1000, batch_size=train_amount, verbose=1)

# Final evaluation of the model
print(model.summary())

scores = model.evaluate(x_test, y_test, verbose=0)
print("Model 1: Accuracy (test data): " + str(scores[1]))

#plt.plot(history.history['mean_squared_error'])
#plt.show()

# Save model
path = './models/saved_model.h5'

model.save(path)
print("Saved to: " + path)

# Plot results

time = arange(0,len(x_test_scaled))

plot_data1 = model.predict(x_test_scaled)
plot_data2 = y_test_scaled

plot_data1 = scY.inverse_transform(plot_data1)
plot_data2 = scY.inverse_transform(plot_data2)
plt.plot(time, plot_data2, time, plot_data1)

plt.ylabel('refugees')
plt.show()
