#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:21:14 2018

@author: kirktsui
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np



iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


#Use GridSearchCV to Build an Optimal Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)
param_grid = [{'criterion':['gini'],'max_depth':np.arange(1,7)},
                   {'criterion':['entropy'],'max_depth':np.arange(1,7)}]
gs = GridSearchCV(DecisionTreeClassifier(), 
                      param_grid = param_grid, cv=10, n_jobs = -1)
    
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

bgs=gs.best_estimator_


#Manual score check
in_sample_scores=[]
out_sample_scores =[]
for rs in np.arange(1,11):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rs, stratify=y)
    bgs.fit(X_train, y_train)
    
    in_sample_scores.append(bgs.score(X_train, y_train))
    out_sample_scores.append( bgs.score(X_test, y_test))
    
print("\nin-sample scores:%s\n\nout-of-sample scores:%s\n"%(in_sample_scores,out_sample_scores))
print("in-sample accuracy =%.3f +- %.3f"% (np.mean(in_sample_scores),np.std(in_sample_scores)))
print("out-sample accuracy =%.3f +- %.3f"%( np.mean(out_sample_scores),np.std(out_sample_scores)))




#Cross Validation
scores =[]

scores = cross_val_score(estimator = bgs, X=X_train, y= y_train,cv =10, n_jobs = 1)

print('CV scores:%s'%scores)
print('CV accuracy: %.3f +/- %.3f\n' % (np.mean(scores), np.std(scores)))

print('Out-of-sample score: %.3f'%bgs.score(X_test, y_test))
  
print("My name is Jianhao Cui")
print("My NetID is: jianhao3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




