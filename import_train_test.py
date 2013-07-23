
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


#Find number of classes;
classes_full = []
raw_data = []
full_dim = 0
import csv



import numpy as np
from numpy import genfromtxt
from sklearn import (metrics, cross_validation, linear_model, preprocessing, svm)
my_data = genfromtxt('/hdd/kaggle/amazon_perms/train.csv', delimiter=",")
my_data_TT = genfromtxt('/hdd/kaggle/amazon_perms/test.csv', delimiter=",")
new_array = my_data[1:]
new_array_TT = my_data_TT[1:]


subset = [1, 2, 4, 5, 7, 8, 9]

y, X = new_array[:,0], new_array[:, subset]


#y, X = training_data[:,0], training_data[:, subset]
#y_test, X_test = 0, new_array_TT[:, subset]

encoder = preprocessing.OneHotEncoder()
encoder.fit((X))
training_data = encoder.transform(X)  # Returns a sparse matrix (see numpy.sparse)



X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(training_data, y, test_size=.25, random_state=26)

print np.size(X_train, 0)
print np.size(X_cv, 0)

alldata = ClassificationDataSet(np.size(X_train, 1), 1, nb_classes=2)
valdata = ClassificationDataSet(np.size(X_cv, 1), 1, nb_classes=2)
#X_test = encoder.transform(X_test)

for i in range(len(y_train)):
    print X_train[i, :]
    print alldata.addSample(X_train[i, :], y_train[i])

print training_data[0, 1]

val_actual = []
for row in test_data:
    valdata.addSample(row[1:], row[0])
    val_actual.append(row[0])

valdata._convertToOneOfMany()

tstdata, trndata = alldata.splitWithProportion( 0.25 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork( trndata.indim, 100, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)\

trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
tstresult = percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['class'] )

import numpy as NP
from sklearn import datasets
from sklearn import datasets as DS
D = digits.data
T = digits.target

from sklearn import metrics

for i in range(20):
    trainer.trainEpochs( 5 )
    print "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
    out = fnn.activateOnDataset(valdata)
    out = out.argmax(axis=1)
    print min(out)
    fpr, tpr, thresholds = metrics.roc_curve(val_actual, out, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    print auc