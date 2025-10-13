# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:22:01 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.svm import SVC



iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)","petal width (cm)"]].values
y =  ( iris.target == 2)

svm_clf = make_pipeline( StandardScaler(), LinearSVC(C=1, random_state=42) )
svm_clf.fit(X,y)

X_new = [ [ 5.5,1.7 ], [5.0, 1.5] ]
svm_clf.predict(X_new)

svm_clf.decision_function(X_new)

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynominal_svm_clf = make_pipeline( PolynomialFeatures( degree=3 ), StandardScaler(), LinearSVC( C=10, max_iter=10_000, random_state=42 )  )

polynominal_svm_clf.fit(X, y)

poly_kernel_svm_clf = make_pipeline( StandardScaler(), SVC( kernel="poly", degree=3, coef0=1, C=5 ) )
poly_kernel_svm_clf.fit(X, y)

rbf_kernel_svm_clf = make_pipeline( StandardScaler(), SVC(kernel="rbf", gamma=5, C=0.001) )
rbf_kernel_svm_clf.fit(X, y)



















































