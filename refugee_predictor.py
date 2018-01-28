# Based on https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://medium.com/@pushkarmandot/build-your-first-deep-learning-neural-network-model-using-keras-in-python-a90b5864116d

# Expecting CSV data in format:
# Country (string), avg, max, min, refugees, timestep (0 to train amount)

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
import matplotlib.patches as mpatches

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

from datetime import datetime

def get_country_data(x,y,country_encoder,country_name):
    filtered_x = []
    filtered_y = []

    for i in range(len(x)):
        index = int(x[i][0])
        if (country_encoder[index] == country_name):
            filtered_x.append(x[i])
            filtered_y.append(y[i])

    return asarray(filtered_x), asarray(filtered_y)

def save_plot(model, x, y, scaler_x, scaler_y, disaster_encoder, path, forecast, values_per_country):
    x_scaled = scaler_x.transform(x)
    expected = y

    predicted = model.predict(x_scaled)   
    predicted = scaler_y.inverse_transform(predicted)

    predicted_partial = []
    expected_partial = []

    start = 0
    end = len(x)

    for i in arange(start, end):
        if ((i - start) % forecast == 0):
            predicted_partial.append(predicted[i - start])
            expected_partial.append(expected[i - start])    

    plot_data_predicted = [num for elem in predicted_partial for num in elem]
    plot_data_expected = [num for elem in expected_partial for num in elem]

    end = len(plot_data_expected)

    # Set figure size
    fig_size = plt.rcParams["figure.figsize"]    
    fig_size[0] = 20
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size

    # Plot
    fig, ax1 = plt.subplots()

    left, bottom, width, height = [0.125, -0.1, 0.776, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])

    time = arange(start, end)
    ax1.plot(time, plot_data_predicted, 'g--', label='Predicted')

    time = arange(start, end)
    ax1.plot(time, plot_data_expected, 'tab:orange', label='Real Data')

    time = arange(start,end,forecast)
    ax1.plot(time,[plot_data_predicted[i - start] for i in arange(start,end,forecast)], 'go', label='Start Point for Forecast')
    
    ax1.set_xlabel('Time')
    ax1.legend(loc='upper left')

    end = len(x)
    time = arange(start, end)

    avg_temps = [elem[1] for elem in x]
    max_temps = [elem[2] for elem in x]
    min_temps = [elem[3] for elem in x]
    ax2.plot(time, max_temps, 'r-', label='Maxmimum')
    ax2.plot(time, avg_temps, 'g-', label='Average')
    ax2.plot(time, min_temps, 'b-', label='Minimum')

    disasters_x = []
    disasters_y = []

    disaster_y_pos = min(min_temps) - 5

    for i in range(0, len(x)):
        dis_index = int (x[i][4])

        if (disaster_encoder[dis_index] != '*'):
            disasters_x.append(i)
            disasters_y.append(disaster_y_pos)

    ax2.plot(disasters_x, disasters_y, 'ro', label='Disasters')

    ax2.legend(loc='upper left')

    ax1.set_ylabel('Refugees', color='C1')
    ax2.set_ylabel('Temperature', color='b')

    left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]

    # Save figure
    # plt.show()
    plt.savefig(path + '.svg', dpi=500, additional_artists=ax2, bbox_inches="tight", format="svg", transparent=True)
    print ("Graph saved to " + path +  '.svg')
    plt.close()

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

    csvfile.close()

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
    return (x_final, y_final, country_encoder, disaster_encoder)

def split_data(x, y, lookback, forecast, train_amount,vpc):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

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

def create_model(input_dimension,lookback,forecast):
    # create model
    model = Sequential()
    model.add(Dense(units=4 + lookback * 4, kernel_initializer='random_uniform',
                activation='relu', input_dim=input_dimension))
    model.add(Dense(units=4 + lookback * 3,
                kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(units=4 + lookback * 2,
                kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(units=4 + lookback,
                kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(units=forecast, kernel_initializer='random_uniform', activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    return model

def get_scalers(x_train, y_train):
    scX = StandardScaler()
    scY = MinMaxScaler()

    scX.fit(x_train)
    scY.fit(y_train)

    return (scX, scY)

def train_model(model,x,y,l,f,n,s,indi,trainAmount,vpc):
    # split dataset
    x_train, y_train, x_test, y_test = split_data(x, y, l, f, trainAmount,vpc)

    scX, scY = get_scalers(x_train, y_train)

    # scale values
    x_train_scaled = scX.transform(x_train)
    x_test_scaled = scX.transform(x_test)
    y_train_scaled = scY.transform(y_train)
    y_test_scaled = scY.transform(y_test)

    x_train = x_train_scaled
    x_test = x_test_scaled
    y_train = y_train_scaled
    y_test = y_test_scaled

    for i in range(1, s + 1):
        model.fit(x_train, y_train, epochs=n, batch_size=trainAmount, verbose=2)

    return model