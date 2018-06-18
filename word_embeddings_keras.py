'''
Created on Apr 17, 2018
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
@author: kaliaanup
'''
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import (Embedding, Input, LSTM, RepeatVector, Dropout, Dense, Flatten)

#other models for 
def gen_word_embeddings_keras_example_1():
    #Embedding(input_dim,output_dim, input_length)
    #1.input_dim-size of vocab for text data
    #2.output_dim-size of vector space in which words will be embedded
    #embedding layer is a 2D vector with one embedding for each word in the input sequence of words
    e = Embedding(200, 32, input_length=50)
    docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
    labels = np.array([1,1,1,1,1,0,0,0,0,0])
    # integer encode the documents
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    print(encoded_docs)
    
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
# compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
    print(model.summary())
    
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    
#other models for 
#using glove
def gen_word_embeddings_keras_example_2():
        # define documents
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']
    # define class labels
    labels = np.array([1,1,1,1,1,0,0,0,0,0])
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('word_vectors/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # define model
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    
if __name__ == '__main__':
    #gen_word_embeddings_keras_example_1()
    gen_word_embeddings_keras_example_2()