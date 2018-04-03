'''
Created on Mar 24, 2018

@author: kaliaanup
'''
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout

nb_classes = 10


# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# 
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), 
#               metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Put everything on grayscale
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train)

print X_train[0].shape
plt.imshow(X_train[0].reshape(28, 28))


print np.asarray(range(10))
print Y_train[0].astype('int')

plt.imshow(X_val[0].reshape(28, 28))

#network_history = model.fit(X_train, Y_train, batch_size=128, epochs=2, verbose=1, validation_data=(X_val, Y_val))

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['acc'])
    plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

#plot_history(network_history)

early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer=SGD(), 
              metrics=['accuracy'])
     
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=100, 
          batch_size=128, verbose=True, callbacks=[early_stop])

#
#Extract hidden layer representation of the given data


model_truncated = Sequential()
model_truncated.add(Dense(512, activation='relu', input_shape=(784,)))
model_truncated.add(Dropout(0.2))
model_truncated.add(Dense(512, activation='relu'))

for i, layer in enumerate(model_truncated.layers):
    layer.set_weights(model.layers[i].get_weights())

model_truncated.compile(loss='categorical_crossentropy', optimizer=SGD(), 
              metrics=['accuracy'])
