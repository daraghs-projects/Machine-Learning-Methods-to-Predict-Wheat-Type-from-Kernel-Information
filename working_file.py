# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:18:01 2024

@author: dobme
"""

'''
importing all necessary packages
'''

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
from sklearn import set_config
import pickle
import matplotlib.pyplot as plt

'''
Importing data and checking for na values.
'''

df = pd.read_csv("seeds.csv")
df.head()

df.isna().sum()

'''
There is no na values. The next step is to check for outliers, this will
be done with boxplots and histograms
'''

df.loc[:,'area':'length of kernel groove'].boxplot(figsize=(20,5))
plt.savefig("outliers_boxplot.png")
plt.show()

df.loc[:,'area':'length of kernel groove'].hist(bins=10, figsize=(25, 20))
plt.savefig("outliers_hist.png")
plt.show()

'''
Area, perimeter, length of kernel and length of kernel groove appear to be
left skewed and could benefit from a log transform.

The next step is to prepare the data for modelling, this will be done by
converting the dependant variable (type) to binary and splitting the data
into two seperate components so it can be modelled.
'''

mapper = {1: 1, 2: 0, 3: 0}
df['class'] = df['type'].replace(mapper)
df['class'].value_counts()
df = df.drop('type', axis = 1)

y = df['class']
X = df.drop('class', axis=1)

'''
Another important component to this preparation is the log transformation
mentioned earlier. Robust scaler will also be used to prepare the data for
modelling, this subtracts the median and divides by the interquartile range.
The importance of this is to make sure all data is on the same scale.

These preprocessing techniques are used as they may improve the performance
of machine learning models.

The preprocessing pipelines will be prepared next and the data will be divided
into train/test sets.
'''

columns_left_skew = ['area', 'perimeter', 'length of kernel', 'length of kernel groove']

columns_other = [item for item in list(X.columns) 
                             if item not in columns_left_skew]

columns_left_skew_pipeline = Pipeline(
    steps = [
        ("log_transform", FunctionTransformer(np.log)), 
        ("scaler", RobustScaler())
    ]
)

columns_other_pipeline = Pipeline(
    steps = [
        ("scaler", RobustScaler())
    ]
)

preprocess_pipeline = ColumnTransformer(
    transformers = [
        ("left_skew", columns_left_skew_pipeline, columns_left_skew),
        ("other", columns_other_pipeline, columns_other)
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

'''
Now the modelling will begin, three models will be compared; SVM, random forest and Knn.
For each model, the pipeline will be built, then a hyperparameter grid will be created so the best
set of hyperparameters can be searched. Finally the best hyperparameters will be saved for use on the test data.
Note to avoid confusion - there are two types of test data. The first is implemented when cross validation
is used on the training set. The second is a randomly selected set of 20% that will be used at the end
to compare models.
'''

'''
SVM
'''

# create the pipeline
pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('svm', svm.SVC(probability=True))])

set_config(display="diagram")
pipe

# prepare a hyperparameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],  
    'svm__gamma': [1, 0.1, 0.01, 0.001], 
    'svm__kernel': ['rbf', 'linear', 'poly']}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, refit=True)
search.fit(X_train, y_train) #training happens here! SVM is trained 48x5 = 240 times

print("Best CV score = %0.3f" % search.best_score_)
print("Best hyperparameters: ", search.best_params_)

# store the best params and best model for later use
SVM_best_params = search.best_params_
SVM_best_model = search.best_estimator_

'''
Random Forest
'''

# create the pipeline
pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('rf', RandomForestClassifier())])

set_config(display="diagram")
pipe

# prepare a hyperparameter grid
# note that __ can be used to specify the name of a hyperparameter for a specific element in a pipeline
# note also that this is not an exhaustive list of the hyperparameters of RandomForestClassifier and their possible values
param_grid = {
    'rf__n_estimators' : [10,20,30],
    'rf__max_depth': [2, 4, 6, 8]
}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, refit=True)
search.fit(X_train, y_train)
print("Best CV score = %0.3f" % search.best_score_)
print("Best hyperparameters: ", search.best_params_)

# store the best params and best model for later use
RF_best_params = search.best_params_
RF_best_model = search.best_estimator_

'''
Knn
'''

# create the pipeline
pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('knn', KNeighborsClassifier())])

#visualise pipeline
set_config(display="diagram")
pipe

param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=5, refit=True)
search.fit(X_train, y_train)
print("Best CV score = %0.3f" % search.best_score_)
print("Best hyperparameters: ", search.best_params_)

# store the best params and best model for later use
kNN_best_params = search.best_params_
kNN_best_model = search.best_estimator_

'''
Now the models will be compared on the test set.
Evaluation metrics will be accuracy, precision, recall, f1 score
and auc score.
'''

mean_fpr = np.linspace(start=0, stop=1, num=100)

# model - a trained binary probabilistic classification model;
#         it is assumed that there are two classes: 0 and 1
#         and the classifier learns to predict probabilities for the examples to belong to class 1

def evaluate_model(X_test, y_test, model):
    
    # compute probabilistic predictiond for the evaluation set
    _probabilities = model.predict_proba(X_test)[:, 1]
    
    # compute exact predictiond for the evaluation set
    _predicted_values = model.predict(X_test)
        
    # compute accuracy
    _accuracy = accuracy_score(y_test, _predicted_values)
        
    # compute precision, recall and f1 score for class 1
    _precision, _recall, _f1_score, _ = precision_recall_fscore_support(y_test, _predicted_values, labels=[1])
    
    # compute fpr and tpr values for various thresholds 
    # by comparing the true target values to the predicted probabilities for class 1
    _fpr, _tpr, _ = roc_curve(y_test, _probabilities)
        
    # compute true positive rates for the values in the array mean_fpr
    _tpr_transformed = np.array([np.interp(mean_fpr, _fpr, _tpr)])
    
    # compute the area under the curve
    _auc = auc(_fpr, _tpr)
            
    return _accuracy, _precision[0], _recall[0], _f1_score[0], _tpr_transformed, _auc

SVM_accuracy, SVM_precision, SVM_recall, SVM_f1_score, SVM_tpr, SVM_auc = evaluate_model(X_test, y_test, SVM_best_model)
RF_accuracy, RF_precision, RF_recall, RF_f1_score, RF_tpr, RF_auc = evaluate_model(X_test, y_test, RF_best_model)
kNN_accuracy, kNN_precision, kNN_recall, kNN_f1_score, kNN_tpr, kNN_auc = evaluate_model(X_test, y_test, kNN_best_model)

# plot display results comparison of accuracy, precision, recall, and f1 score
SVM_metrics = np.array([SVM_accuracy, SVM_precision, SVM_recall, SVM_f1_score])
RF_metrics = np.array([RF_accuracy, RF_precision, RF_recall, RF_f1_score])
kNN_metrics = np.array([kNN_accuracy, kNN_precision, kNN_recall, kNN_f1_score])
index = ['accuracy', 'precision', 'recall', 'F1-score']
df_metrics = pd.DataFrame({'SVM': SVM_metrics, 'Random Forest': RF_metrics,  'kNN': kNN_metrics}, index=index)
df_metrics.plot.bar(rot=0)
plt.legend(loc="lower right")
plt.savefig("results.bar.png")
plt.show()

# plot display results comparison of roc curves with auc values
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
plt.plot(mean_fpr, SVM_tpr[0,:], lw=2, color='blue', label='SVM (AUC = %0.4f)' % (SVM_auc), alpha=0.8)
plt.plot(mean_fpr, RF_tpr[0,:], lw=2, color='orange', label='Random Forest (AUC = %0.4f)' % (RF_auc), alpha=0.8)
plt.plot(mean_fpr, kNN_tpr[0,:], lw=2, color='orange', label='kNN (AUC = %0.4f)' % (kNN_auc), alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for multiple classifiers')
plt.legend(loc="lower right")
plt.savefig("results.roc.png")
plt.show()




















