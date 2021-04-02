import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.metrics import classification_report

import os

# scikit-learn k-fold cross-validation
# from numpy import array
from sklearn.model_selection import KFold

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

data = pd.concat([train_df, test_df], ignore_index=True).values

# breakpoint()
parameters = {
    'penalty' : ['l2'],
    'C': [6],
    'solver': ['liblinear','lbfgs']
             }

kfold_acc = []
kfold = KFold(3, True, 1)
for train, test in kfold.split(data):
	# print('train: %s, test: %s' % (data[train], data[test]))
    X_train = data[train][:, 0]
    y_train = data[train][:, 1]
    X_test = data[test][:, 0]
    y_test = data[test][:, 1]
    # X_test = test_df['instruction']
    # y_test = test_df['target']

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # LOGISTIC REGRESSION
    logreg = LogisticRegression(max_iter=200, class_weight='balanced', warm_start=True)
    logreg = GridSearchCV(logreg, parameters, cv=5, verbose=5, n_jobs=-1)

    logreg.fit(X_train, y_train)
    lr_prediction = logreg.predict(X_test)
    lr_pred_proba = logreg.predict_proba(X_test)

    correct = sum(lr_prediction == y_test)
    total = len(y_test)
    acc = correct/total
    print(acc)
    kfold_acc.append(acc)
    print('best_params_', logreg.best_params_)


    print(classification_report(y_test, lr_prediction))

# print ('Training Results')
# #print (logreg.score(X_train, y_train.values.ravel()))
# print ("recall: "+ str(recall_score(y_train, logreg.predict(X_train))))
# print ("accuracy: "+ str(metrics.accuracy_score(y_train,logreg.predict(X_train))))
# print ("roc_auc_score: "+ str(roc_auc_score(y_train,logreg.predict(X_train))))

# print ('\nTest Results')
# #print (logreg.score(X_test, y_test.values.ravel()))
# print ("recall: "+ str(recall_score(y_test, logreg.predict(X_test))))
# print ("accuracy: "+ str(metrics.accuracy_score(y_test,logreg.predict(X_test))))
# print ("roc_auc_score: "+ str(roc_auc_score(y_test,logreg.predict(X_test))))

# print ('\nScore Results')
# #print (logreg.score(X_score, y_score.values.ravel()))
# print ("recall: "+ str(recall_score(y_score.values.ravel(), logreg.predict(X_score))))
# print ("accuracy: "+ str(metrics.accuracy_score(y_score,logreg.predict(X_score))))
# print ("roc_auc_score: "+ str(roc_auc_score(y_score,logreg.predict(X_score))))
print('Avg acc: ', sum(kfold_acc)/len(kfold_acc))