#!/usr/bin/python

from numpy import *
from nltk import ngrams
from string import printable
from multiprocessing import Pool
import random
SAMPLE_RATE = 0.001

def ngrams_extract(string):
    if random.random() < SAMPLE_RATE:
        print '[*]',string
    l = list
    grams = l(ngrams(string,2)) + l(ngrams(string,3)) + l(ngrams(string,4)) + l(ngrams(string,5))
    SIZE = 1024
    vec = zeros((SIZE,))
    for t in grams:
        vec[hash(t)%SIZE]+=1
    return log(vec+1.0)

def sequence(string,maxlen=100):
    tokens = []
    for c in string:
        if not c in printable:
            continue
        tokens.append(printable.index(c))
    tokens = tokens[-maxlen:]
    if len(tokens) < maxlen:
        tokens = [0]*(maxlen-len(tokens))+tokens
    return tokens

def parallel_extract(strings,func):
    pool = Pool()
    feats = pool.map(func,strings)
    feats = array(feats)
    pool.close()
    return feats

def extract_split(strings,y,splitidx):
    splitidx = splitidx * len(y)
    splitidx = int(splitidx)
    strings = parallel_extract(strings,func)
    X_train,X_test = strings[:splitidx],strings[splitidx:]
    y_train,y_test = y[:splitidx],strings[splitidx:]
    return (X_train,X_test),(y_train,y_test)
