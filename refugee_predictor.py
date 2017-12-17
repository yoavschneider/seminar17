# Based on https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# Expecting CSV data in format:
# Country (string), avg, max, min, refugees, timestep (0 to 443)

import csv

import numpy
from numpy import array
from numpy import asarray
from numpy import argmax
from numpy import unique

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

number_of_values_per_country = 444

train_amount = 420
test_amount = 24

train_range = range(0, train_amount)
test_range = range(train_amount, train_amount + test_amount)

# Using data from previous months (up to 24)
lookback = 12

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
    x_test = zip (countries_encoded_test, x_test)
    y_train = zip (countries_encoded_train, y_train)
    y_test = zip (countries_encoded_test, y_test)

    x_train_final = []
    x_test_final = []
    y_train_final = []
    y_test_final = []

    # format now (country, time, avg, max, min, refugees)
    # convert to [country] + [avg,max,min] + lookback * [avg,max,min,refugees]

    last_values = []

    for (country, (time, avg, max, min, refugees)) in x_train:
        if (time == 0):
            del last_values[:]
        if (time >= lookback):
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

    for (country, (time, avg, max, min, refugees)) in x_test:
        if (time - train_amount == 0):
            del last_values[:]
        if (time - train_amount >= lookback):
            x_test_final.append([country,avg,max,min] + last_values)
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

    for (country, (time, refugees)) in y_test:
        if (time - train_amount >= lookback):
            y_test_final.append(refugees)

    x_train = asarray(x_train_final)
    y_train = asarray(y_train_final)
    x_test = asarray(x_test_final)
    y_test = asarray(y_test_final)

    return ((x_train, y_train), (x_test, y_test))

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
(x_train, y_train), (x_test, y_test) = load_data("LESS_DATA_CORRECTED_17_12.csv")

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)

input_dimension = 4 + lookback * 4

# create the model
model = Sequential()
model.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = input_dimension))
model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())

# Each batch should be one country
model.fit(x_train, y_train, epochs=200, batch_size=train_amount)

# Final evaluation of the model
scores = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy (training data): %.2f%%" % (scores[1]*100))
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy (test data): %.2f%%" % (scores[1]*100))

