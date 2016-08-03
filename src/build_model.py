#!/usr/bin/python

import argparse
import sqlite3
import numpy as np
import random
import os
import datetime
import cPickle
import json
from sklearn.metrics import roc_curve, auc
from modeling import features
from modeling import models
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import defaultdict as ddict

RESULTS_BASE = "results"

args = argparse.ArgumentParser()
args.add_argument("dbfile",help="Database file to target")
args.add_argument("--max_samples",help="Maximum number of samples to randomly sample from database",type=int,default=2000000)
args.add_argument("--split_method",help="Method by which to split train / test: {entity,unique}",default="unique")
args.add_argument("--model",help="Model to use: {convnet,rf,mlp}",default="mlp")
args.add_argument("--split_ratio",help="Split ratio",default=0.8,type=float)
args.add_argument("--val_dbfile",help="Validation database",default=None)
args.add_argument("--max_val_samples",help="Maximum validation samples",default=10000)
args.add_argument("--results_dir",help="Name of results directory",default=None)
args = args.parse_args()

def get_training_data(dbfile,V_dbfile=None):
    """
    This function gets training data from a database returns training and test data
    Args:
        dbfile: the base SQLite db file with the training data
        V_dbfile: an optional held out validation SQLite db file
    """
    db = sqlite3.connect(dbfile)
    cursor = db.cursor()
    if V_dbfile:
        V_db = sqlite3.connect(V_dbfile)
        V_cursor = V_db.cursor()
    split_method = args.split_method
    query = ""
    if split_method == 'entity':
        cursor.execute("""
        select raw, entity_id, label
        from string
        join entity on entity_id = entity.id
        join uniquestring on uniquestring_id = uniquestring.id
        """)
        data = list(cursor.fetchall())[:args.max_samples]
        entity_strings = ddict(list)
        for raw, eid, label in data:
            if len(entity_strings[(eid,label)]) < 100:
                entity_strings[(eid,label)].append(raw)
        entity_strings = entity_strings.items()
        random.shuffle(entity_strings)
        datalen = len(entity_strings)
        splitidx = int(args.split_ratio * datalen)

        train_strings,test_strings=entity_strings[:splitidx],entity_strings[splitidx:]
        train_X = []
        train_y = []
        test_X = []
        test_y = []

        for (eid,label),strings in train_strings:
            for string in strings:
                train_X.append(string)
                train_y.append(int(label))

        for (eid,label),strings in test_strings:
            for string in strings:
                test_X.append(string)
                test_y.append(int(label))
        
        return train_X,test_X,train_y,test_y
    elif split_method == 'unique':
        def get_unique_data(target_cursor,max_samples,skipset=None):
            target_cursor.execute("""
            select raw, p_malware
            from uniquestring 
            """)
            strings = []
            labels = []
            rows = list(target_cursor.fetchall())
            random.shuffle(rows)

            for string, p_malware in rows:
                string = string.lower()
                if skipset and string in skipset:
                    print "SKIPPING:",string
                    continue
                if p_malware == 1.0:
                    labels.append(1)
                elif p_malware < 1.0:
                    labels.append(0)
                strings.append(string) 
                if len(strings) == max_samples:
                    break
            return strings,labels
        strings,labels = get_unique_data(cursor,args.max_samples)
        val_skipset = set(strings)

        if V_dbfile:
            V_vec,V_labels = get_unique_data(V_cursor,args.max_val_samples,val_skipset)

        datalen = len(labels)
        splitidx = int(args.split_ratio * datalen)
        train_strings,test_strings=strings[:splitidx],strings[splitidx:]
        train_labels,test_labels=labels[:splitidx],labels[splitidx:]

        if V_dbfile:
            return train_strings,test_strings,train_labels,test_labels,V_vec,V_labels
        else:
            return train_strings,test_strings,train_labels,test_labels

def get_features(samples):
    """
    Get features to feed the model.  If the model is random forest or mlp, these are n-grams.  Otherwise it's a matrix
    representation of the raw string

    Args:
        samples: a list of lists of integer character values

    Returns:
        model, scores for test data
        OR
        model, scores for test data, scores for validation data
    """
    if args.model in ('rf','mlp'):
        feats = features.parallel_extract(samples,features.ngrams_extract)
    else:
        feats = features.parallel_extract(samples,features.sequence)
    return np.array(feats)

def fit_model(X_train,y_train,X_test,V_vec=None):
    """
    Fit the model to data

    Args:
        X_train: the observation matrix for training
        y_train: the label vector
        X_test: what we'll predict on
    Returns:
        The scores for X_test
    """
    if args.model == 'rf':
        rf = models.randomforest()
        rf.fit(X_train,y_train)
        probas = rf.predict_proba(X_test)[:,-1]
        if V_vec != None:
            V_probas = rf.predict_proba(V_vec)[:,-1]
            return rf,probas,V_probas
        else:
            return rf,probas
    elif args.model in ('mlp','convnet'):
        if args.model == 'mlp':
            model = models.mlp_model()
        elif args.model == 'convnet':
            model = models.bag_of_convs_model() 
        early_stop = models.EarlyStopping(patience=10)
        val_split = int(0.95*len(X_train))
        final_X_train,final_y_train = X_train[:val_split],y_train[:val_split]
        X_val,y_val = X_train[val_split:],y_train[val_split:]
        print "Fitting model"
        model.fit(final_X_train,final_y_train,batch_size=1024,
            nb_epoch=200,
            callbacks=[early_stop],
            validation_data=(X_val,y_val))
        print "Predicting probability"
        probas = model.predict(X_test)
        if V_vec != None:
            V_probas = model.predict(V_vec)
            return model,probas,V_probas
        else:
            return model,probas

def get_experiment_string():
    """
    Get a string that uniquely identifies and helpfully describes this particular experimental run.  We'll use this
    string as the name of the directory where we'll store model weights and the test results

    Returns: 
        Experiment descriptor string
    """
    d = str(datetime.datetime.utcnow())
    d = d.replace(" ","T")
    d = d + "_"
    for k,v in args.__dict__.items():
        d+=".{0}-{1}".format(str(k),str(v))
    return d

def evaluate_model(model,y_pred,y_test,test_strings,V_probas=None,V_labels=None):
    """
    Compute a ROC curve for the experimental run and store the result in the results directory

    Args:
        model: the model to be evaluated
        y_pred: the scores for the test data
        y_test: the labels for the test data
        test_strings: the input strings for the test data
        V_probas: the predictions for the validation set
        V_labels: the ground truth for the validation set
    """
    fpr,tpr,thres = roc_curve(y_test,y_pred)
    plt.plot(fpr,tpr,'r-',label="Test set")
    if V_labels != None:
        fpr,tpr,thres = roc_curve(V_labels,V_probas)
        plt.plot(fpr,tpr,'b-',label="Validation set")
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.xlim([0,1])
    plt.ylim([0,1])
    os.chdir(RESULTS_BASE)
    if not args.results_dir:
        subdir = get_experiment_string()
    else:
        subdir = args.results_dir
    os.mkdir(subdir)
    os.chdir(subdir)
    plt.savefig("roc.png")
    sample_scores = []
    for spred,struth,sstring in zip(y_pred,y_test,test_strings)[:5000]:
        try:
            spred = spred[0]
        except:
            pass
        print spred,struth,sstring
        sample_scores.append({
            'pred':float(spred),
            'truth':int(struth),
            'string':str(sstring)
        })
    json.dump(sample_scores,open("sample_scores.json","w+"),indent=2)

    try:
        model_json = model.to_json()
        model.save_weights("model_weights.h5")
        ofile = open("model_def.json","w+")
        ofile.write(model_json)
        ofile.close()
    except:
        cPickle.dump(model,open("model.pkl","w+"),2)

    cPickle.dump((fpr,tpr,thres),
        open("fpr-tpr-thres.pkl","w+"))

if not args.val_dbfile:
    X_train,X_test,y_train,y_test = get_training_data(args.dbfile)
    X_train = get_features(X_train)
    test_strings = np.array(X_test)
    X_test = get_features(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print "Array lengths", len(X_train),len(X_test),len(y_train),len(y_test)
    model, probas = fit_model(X_train,y_train,X_test)
    print "Evaluating model"
    evaluate_model(model,probas,y_test,test_strings)
else:
    X_train,X_test,y_train,y_test,V_vec,V_labels = get_training_data(
        args.dbfile,
        args.val_dbfile
    )
    X_train = get_features(X_train)
    V_vec = get_features(V_vec)
    V_labels = np.array(V_labels)
    test_strings = np.array(X_test)
    X_test = get_features(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print "Array lengths", len(X_train),len(X_test),len(y_train),len(y_test)
    model, probas, V_probas = fit_model(X_train,y_train,X_test,V_vec)
    print "Evaluating model"
    evaluate_model(model,probas,y_test,test_strings,V_probas,V_labels)
