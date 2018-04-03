'''
Created on Mar 24, 2018

@author: kaliaanup
'''
from kaggle_data import load_data, preprocess_data, preprocess_labels
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

X_train, labels = load_data('data/kaggle_ottogroup/train.csv', train=True)
X_train, scaler = preprocess_data(X_train)
Y_train, encoder = preprocess_labels(labels)

X_test, ids = load_data('data/kaggle_ottogroup/test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = Y_train.shape[1]
print(nb_classes, 'classes')

dims = X_train.shape[1]
print(dims, 'dims')

print np.unique(labels)


dims = X_train.shape[1]
print(dims, 'dims')
print("Building model...")

nb_classes = Y_train.shape[1]
print(nb_classes, 'classes')

model = Sequential()
model.add(Dense(nb_classes, input_shape=(dims,), activation='sigmoid'))
#add more layers

model.add(Activation('softmax'))

#add optimizer stochastic gradient descent
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(X_train, Y_train)

model.summary()

# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)
# 
# fBestModel = 'best_model.h5' 
# early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1) 
# best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)
# 
# model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=50, 
#           batch_size=128, verbose=True, callbacks=[best_model, early_stop])
