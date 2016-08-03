#!/usr/bin/python

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten, Lambda, Merge
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.callbacks import EarlyStopping
from collections import defaultdict as ddict
from sklearn.ensemble import RandomForestClassifier as RF
from numpy import *

def sum_1d(X):
    return K.sum(X, axis=1)

def getconvmodel(filter_length,nb_filter):
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filter,
                            input_shape=(100,32),
                            filter_length=filter_length,
                            border_mode='same',
                            activation='relu',
                            subsample_length=1))
    model.add(Lambda(sum_1d, output_shape=(nb_filter,)))
    #model.add(BatchNormalization(mode=0))
    model.add(Dropout(0.5))
    return model

def mlp_model(optimizer="adam"):
    main_input = Input(shape=(1024,), dtype='float32', name='main_input')

    middle = Dense(1024,activation='relu')(main_input)
    middle = Dropout(0.5)(middle)
    middle = BatchNormalization()(middle)

    middle = Dense(1024,activation='relu')(main_input)
    middle = Dropout(0.5)(middle)
    middle = BatchNormalization()(middle)

    output = Dense(1,activation='sigmoid')(middle)

    model = Model(input=main_input,output=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def bag_of_convs_model(optimizer="adam",compile=True):
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    embedding = Embedding(output_dim=32, input_dim=100, input_length=100,
        dropout=0)(main_input)

    conv1 = getconvmodel(2,256)(embedding)
    conv2 = getconvmodel(3,256)(embedding)
    conv3 = getconvmodel(4,256)(embedding)
    conv4 = getconvmodel(5,256)(embedding)

    merged = merge([conv1,conv2,conv3,conv4],mode="concat")

    middle = Dense(1024,activation='relu')(merged)
    middle = Dropout(0.5)(middle)

    middle = Dense(1024,activation='relu')(middle)
    middle = Dropout(0.5)(middle)

    output = Dense(1,activation='sigmoid')(middle)

    model = Model(input=main_input,output=output)
    if compile:
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def randomforest():
    rf = RF(256,verbose=True,n_jobs=50)
    return rf
