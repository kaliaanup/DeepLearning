'''
Created on Apr 17, 2018

@author: kaliaanup
'''
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers.recurrent import LSTM

import numpy as np

LSTM_size_1 = 5

data = [
    [ [1 ,3] ],
    [ [2 ,4] ],
    [ [3 ,5 ] ],
    [ [4 ,6 ] ],
    [ [5 ,7 ] ],
    [ [6 ,8 ] ],
    [ [7 ,9 ] ],
    [ [8 ,10 ] ],
    [ [9 ,11 ] ]
]
print data

data = np.array(data)

print data

in_dim = data.shape[-1]
print in_dim
m = Sequential()
m.add(LSTM(LSTM_size_1, input_dim=in_dim, return_sequences=True))
m.add(LSTM(in_dim, return_sequences=True))
m.add(Activation('linear'))
m.compile(loss='mse', optimizer='RMSprop')
m.fit(data,data, nb_epoch=2)

print m.predict(data)
