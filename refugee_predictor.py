# Based on https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

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

train_range = range(0, 420)
test_range = range(420, 444)

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
                x_train.append((int(time), float(avg), float(max), float(min)))
                y_train.append(int(refugees))
                countries_train.append(country)
            if (int(time) in test_range):
                x_test.append((int(time), float(avg), float(max), float(min)))
                y_test.append(int(refugees))
                countries_test.append(country)

    countries_train = array(countries_train)
    countries_test = array(countries_test)

    b1 ,countries_encoded_train = unique(countries_train, return_inverse=True)
    b2 ,countries_encoded_test = unique(countries_test, return_inverse=True)

    x_train = zip (countries_encoded_train, x_train)
    x_test = zip (countries_encoded_test, x_test)

    x_train_joined = []
    x_test_joined = []

    for (a,(b,c,d,e)) in x_train:
        x_train_joined.append((a,b,c,d,e))
    for (a,(b,c,d,e)) in x_test:
        x_test_joined.append((a,b,c,d,e))

    x_train = asarray(x_train_joined)
    y_train = asarray(y_train)
    x_test = asarray(x_test_joined)
    y_test = asarray(y_test)

    return ((x_train, y_train), (x_test, y_test))

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
(x_train, y_train), (x_test, y_test) = load_data("ALL_DATA_CORRECTED_16_12.csv")

print (x_train.shape)
print (x_test.shape)

# create the model
model = Sequential()
'''
model.add(SimpleRNN(input_dim=1, output_dim=50))


model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

print(model.summary())
model.fit(x_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

