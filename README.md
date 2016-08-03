# eXpose Deep Neural Network

Attribution: Invincea Labs, josh.saxe@invincea.com

License: (for non-commercial use) https://creativecommons.org/licenses/by-nc/3.0/us/legalcode

This projects implements a character-level convolutional neural network binary classifier designed to do well at cybersecurity detection problems, specifically problems that involve determining whether short, tweet-length character strings (e.g. file paths, URLs) are malicious or not.  The model takes raw character sequences as its input and outputs a 0-1 suspiciousness score.  Specific problems I have tested the model on are:

1. Detecting malicious URLs based on the URL character string (i.e. the model looks at the URL itself, not the contents of the web page the URL points to).  I have found that given at least 1 million training examples balanced between benign and malicious URLs, the model can achieve a > 90% detection rate at a 0.1% false positive rate on this task.

2. Detecting malicious Windows file paths.  The model achieves something in the ballpark of 80% detection rate at a 0.1% false positive rate on this task, using on the order of a million training examples balanced between malicious and benign examples.

3. Detecting malicious Windows registry key paths.  The model achieves something in the ballpark of a 60% detection rate at a 0.1% false positive rate on this task, using on the order of a million training examples balanced between malicious and benign examples.

## Basic usage

eXpose uses zerorpc to host the neural network models as RPC services.  From the top level project directory, you can run the eXpose model in three different modes: URL detection mode, path detection mode, and registry key path detection mode, as follows:

`python model_server.py ../data/urls`
`python model_server.py ../data/paths`
`python model_server.py ../data/registry`

These commands start a model RPC server and load in trained neural network weights from the data directory such that the neural network will know how to perform detection on the object of interest.  An example client for the model servers is provided in `src/example_model_client.py` for your convenience.  This script tests the URL detection functionality of eXpose.  You can run this test as follows:

`python src/model_server.py data/models/urls/`

`python src/example_model_client.py`

This should generate the following output:

`[['facebook.com', 0.0016783798346295953],
 ['paypal.com-confirm-your-paypal-account.spectragroup-inc.com/',
  0.991065502166748],
 ['paypal.com-confirm-your-paypal-account.josh_made_this_up.com/',
  0.9811802506446838],
 ['paypal.com', 0.2289438247680664]]`
 
## Dependencies (install these with pip)

* keras
* h5py
* nltk
* matplotlib
* scipy
* numpy
* sklearn
* sqlite3
* zerorpc
* peewee

## Training new models

To train a new model you must create a sqlite3 database using the `util/db.py` `makedb()` function.  You then can train a model using the `build_model.py` script provided in the release.  A few things to note about populating the SQLite schema with your data:
* The schema is kept intentionally general, with two basic objects: a `String` and an `Entity`, with a one to many relationship from `Entity` to `String`.
* The idea is that, say, in the case of URLs, the domain prefix to the URL (e.g. `www.microsoft.com`) is distinct from the complete URL string (e.g. `www.microsoft.com/support`).
* The motivation for this schema is that we may wish to ensure that strings from the same `Entity` do not wind up in both the training and test set when validating our model (you can make sure validation splits the data in this way by turning on `entity` mode in the `build_model.py` flags)