# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:42:19 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


voting_clf = VotingClassifier( estimators=[ ( 'lr', LogisticRegression(random_state=42) ),
                                             (  'rf', RandomForestClassifier(random_state=42)),
                                             (  'svc', SVC(random_state=42) ) ] )
voting_clf.fit(X_train, y_train)


for name, clf in voting_clf.named_estimators_.items():
    print( name, "=", clf.score(X_test, y_test) )


voting_clf.predict( X_test[:1] )
print(voting_clf.predict( X_test[:1] ))
print( [ clf.predict(  X_test[:1] )  for clf in voting_clf.estimators_ ] )
voting_clf.score(X_test, y_test)



voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
















