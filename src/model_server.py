#!/usr/bin/python

"""
This script is a simple example of how you can deploy the eXpose models in a client server setup.  We deserialize a user-specified model and then serve it up to the user using the ZeroRPC RPC framework.
"""

import zerorpc
import argparse
import os
import numpy
from modeling import features
from keras.models import model_from_json

parser = argparse.ArgumentParser()
parser.add_argument("model_dir",help="Directory with model files")
parser.add_argument("--model_type",help="Type of model (convnet,mlp)",default="convnet")
args = parser.parse_args()

print "Loading models"
if args.model_type in ('convnet','mlp'):
    model = model_from_json(open(os.path.join(
        args.model_dir,
        "model_def.json"
        )).read())
    model.load_weights(
        os.path.join(
            args.model_dir,
            'model_weights.h5',
        )
    )
    model.compile(optimizer='adam', loss='binary_crossentropy')
else:
    model = cPickle.load(open(
        os.path.join(args.model_dir,'model.pkl')
    ))
print "... done!"

class ModelServer(object):
    def create_input(self,strings):
        """
        Extract features / character encodings from raw data.
        If we're using the random forest or MLP models we get character n-grams.
        If we're using the convnet we get letter ordinal arrays.
        """
        fvecs = []
        for string in strings:
            if args.model_type in ('rf','mlp'):
                fvecs.append(features.ngrams_extract(string))
            elif args.model_type == 'convnet':
                fvecs.append(features.sequence(string))
            else:
                raise Exception("unknown model type")
        return numpy.array(fvecs)

    def score(self, strings):
        """
        Score a list of strings, returning a list [[string,score],...]
        """
        input_data = self.create_input(strings)
        if args.model_type == 'rf':
            result = list(model.predict_probas(input_data)[:,-1])
        elif args.model_type in ('mlp','convnet'):
            result = list(model.predict(input_data).ravel())
        result = map(float,result)
        result = zip(strings,result)
        return result

# set up the server and bind to port 4242
s = zerorpc.Server(ModelServer())
s.bind("tcp://0.0.0.0:4242")
s.run()
