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


#train_amount = 300
#values_per_country = 444


def save_plot(model,index, x, y, scaler_x, scaler_y, path, forecast, start, end):
    x_scaled = scaler_x.transform(x[start:end])
    expected = y[start:end]

    predicted = model.predict(x_scaled)
    predicted = scaler_y.inverse_transform(predicted)

    predicted_partial = []
    expected_partial = []

    for i in arange(start, end):
        if ((i - start) % forecast == 0):
            predicted_partial.append(predicted[i - start])
            expected_partial.append(expected[i - start])
    
    plot_data_predicted = [num for elem in predicted_partial for num in elem]
    plot_data_expected = [num for elem in expected_partial for num in elem]

    time = arange(start, end)
    plt.plot(time, plot_data_predicted)

    time = arange(start, end)
    plt.plot(time, plot_data_expected)

    time = arange(start,end,forecast)
    plt.plot(time,[plot_data_predicted[i - start] for i in arange(start,end,forecast)], 'ro')

    plt.ylabel('refugees')
    # plt.show()
    plt.savefig(path + str(index) + '.jpg', dpi=500)
    print ("Graph saved to " + path + str(index) + '.jpg')
    plt.clf()

def load_data(file, lookback, forecast,vpc):
    x = []
    y = []
    countries = []
    disasters = []

    with open(file, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in rows:
            country, avg, max, min, disaster, refugees, time = row
            x.append((int(time), float(avg), float(
                max), float(min), int(refugees)))
            y.append((int(time), int(refugees)))
            countries.append(country)
            disasters.append(disaster)

    countries = array(countries)
    disasters = array(disasters)

    # Hot one encoding of countries
    country_encoder, countries_encoded = unique(countries, return_inverse=True)
    disaster_encoder, disasters_encoded = unique(
        disasters, return_inverse=True)

    x = zip(countries_encoded, disasters_encoded, x)
    y = zip(countries_encoded, y)

    # format now (country, time, avg, max, min, refugees)
    # convert to [country] + [avg,max,min] + lookback * [avg,max,min,refugees]

    last_values = []
    x_final = []
    y_final = []

    for (country, disaster, (time, avg, max, min, refugees)) in x:
        if (time == 0):
            del last_values[:]

        if (time > vpc - forecast):
            continue

        if (lookback == 0):
            x_final.append([country, avg, max, min, disaster])
        elif (time >= lookback):
            x_final.append([country, avg, max, min, disaster] + last_values)
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

    del last_values[:]

    for (country, (time, refugees)) in y:
        if (time == 0):
            if (last_values):
                y_final.append(last_values.copy())
            del last_values[:]

        if (time < lookback):
            continue

        if (time - lookback > forecast - 1):
            y_final.append(last_values.copy())
            last_values.pop(0)
            last_values.append(refugees)
        else:
            last_values.append(refugees)

    y_final.append(last_values.copy())
    print(x_final, y_final)
    return (x_final, y_final)

def split_data(x, y, lookback, forecast, train_amount,vpc):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    print(str(len(x)) + "," + str(len(y)))

    # First train_amount values are for training (out of 444 months)
    for i in range(0, len(list(x))):
        if (i % (vpc - forecast + 1) < train_amount):
            x_train.append(x[i])
            y_train.append(y[i])
        else:
            x_test.append(x[i])
            y_test.append(y[i])

    x_train = asarray(x_train)
    y_train = asarray(y_train)
    x_test = asarray(x_test)
    y_test = asarray(y_test)

    return (x_train, y_train, x_test, y_test)

# fix random seed for reproducibility
# numpy.random.seed(7)

def train_predictor(l,f,ps,pe,n,s,indi,trainAmount,vpc):

# load the dataset
    (x, y) = load_data("ALL_DATA_2_1.csv", l, f,vpc)
    x_train, y_train, x_test, y_test = split_data(x, y, l, f, trainAmount,vpc)

# scale values
    scX = StandardScaler()
    scY = MinMaxScaler()
    x_train_scaled = scX.fit_transform(x_train)
    x_test_scaled = scX.transform(x_test)

    y_train_scaled = scY.fit_transform(y_train)
    y_test_scaled = scY.transform(y_test)

    x_train = x_train_scaled
    x_test = x_test_scaled
    y_train = y_train_scaled
    y_test = y_test_scaled

    # create model
    model = Sequential()
    model.add(Dense(units=4 + l * 4, kernel_initializer='random_uniform',
                activation='relu', input_dim=indi))
    model.add(Dense(units=4 + l * 3,
                kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(units=4 + l * 2,
                kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(units=4 + l,
                kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(units=f, kernel_initializer='random_uniform', activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return x_train,y_train,x,y,scX,scY


