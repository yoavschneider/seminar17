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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train_amount = 420
values_per_country = 444

##train_range = range(0, train_amount)
#test_range = range(train_amount, train_amount + test_amount)

def save_plot(i, x, y, scaler_x, scaler_y, path):
    time = arange(2850,3064)

    x_scaled = scaler_x.transform(x)

    plot_data_predicted = model.predict(x_scaled[2850:3064])
    plot_data_predicted = scaler_y.inverse_transform(plot_data_predicted)

    plot_data_real = y[2850:3064]

    plt.plot(time, plot_data_real)
    plt.plot(time, plot_data_predicted)

    plt.ylabel('refugees')
    #plt.show()
    plt.savefig(path + str(i) + '.jpg')
    plt.clf()

def load_data(file, lookback):
    x = []
    y = []
    countries = []
    disasters = []

    with open(file, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in rows:
            country, avg, max, min, disaster, refugees, time = row
            x.append((int(time), float(avg), float(max), float(min), int(refugees)))
            y.append((int(time), int(refugees)))
            countries.append(country)
            disasters.append(disaster)

    countries = array(countries)
    disasters = array(disasters)

    # Hot one encoding of countries
    country_encoder ,countries_encoded = unique(countries, return_inverse=True)
    disaster_encoder, disasters_encoded = unique(disasters, return_inverse=True)

    x = zip (countries_encoded, disasters_encoded, x)
    y = zip (countries_encoded, y)

    # format now (country, time, avg, max, min, refugees)
    # convert to [country] + [avg,max,min] + lookback * [avg,max,min,refugees]

    last_values = []
    x_final = []
    y_final = []

    for (country, disaster, (time, avg, max, min, refugees)) in x:
        if (time == 0):
            del last_values[:]
        if (lookback == 0):
            x_final.append([country,avg,max,min,disaster])
        elif (time >= lookback):
            x_final.append([country,avg,max,min,disaster] + last_values)
            last_values.pop(0)
            last_values.pop(0)
            last_values.pop(0)
            last_values.pop(0)
            last_values.pop(0)
            last_values.append(avg)
            last_values.append(max)
            last_values.append(min)
            last_values.append(disaster)
            last_values.append(refugees)
        else:
            last_values.append(avg)
            last_values.append(max)
            last_values.append(min)
            last_values.append(disaster)
            last_values.append(refugees)    

    for (country, (time, refugees)) in y:
        if (time >= lookback):
            y_final.append(refugees)

    return (x_final, y_final)

def split_data(x,y,lookback, train_amount):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # First train_amount values are for training (out of 444 months)
    for i in range(0,len(list(x))):
        if ((i % 444) < train_amount):
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])

    x_train = asarray(x_train)
    y_train = asarray(y_train).reshape(-1,1)
    x_test = asarray(x_test)
    y_test = asarray(y_test).reshape(-1,1)

    return (x_train, y_train, x_test, y_test)

# fix random seed for reproducibility
#numpy.random.seed(7)

lookback = 6

# load the dataset
(x, y) = load_data("ALL_DATA_2_1.csv", lookback)
x_train, y_train, x_test, y_test = split_data(x, y, lookback, train_amount)

# scale values
scX = StandardScaler()
scY = MinMaxScaler()
x_train_scaled = scX.fit_transform(x_train)
x_test_scaled = scX.transform(x_test)

y_train_scaled = scY.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scY.transform(y_test.reshape(-1,1))

input_dimension = 5 + lookback * 5

x_train = x_train_scaled
x_test = x_test_scaled
y_train = y_train_scaled
y_test = y_test_scaled

# create model
model = Sequential()
model.add(Dense(units =  4 + lookback * 4, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = input_dimension))
model.add(Dense(units = 4 + lookback * 3, kernel_initializer = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 4 + lookback * 2, kernel_initializer = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 4 + lookback, kernel_initializer = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'relu'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

# PATH
path = './models/new/model'

# Or load model
model = load_model(path + '.h5')
print(model.summary())

# TRAIN
for i in range(1,10):    
    model.fit(x_train, y_train, epochs=100, batch_size=train_amount, verbose=2)

    # Save model
    model.save(path + '.h5')
    print("Saved to: " + path + '.h5')

    # Plot results
    save_plot(i, x, y, scX, scY, path)
