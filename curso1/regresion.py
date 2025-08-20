# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 07:08:13 2025



@author: jesus
"""

import numpy as np # aplicacion de matematicas
import matplotlib as plt # mostrar datos
import pandas as pd # manipular datos
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Data.csv')

X = dataset.iloc[ :,:-1 ].values
y = dataset.iloc[ :,3 ].values
# axis = 0 columna, axis = 1 fila
imputer = SimpleImputer(missing_values = np.nan, strategy="mean")

imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

labelEncoder_X = LabelEncoder()


X[ :,0 ] = labelEncoder_X.fit_transform(X[ :,0 ])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],    
    remainder='passthrough'                         
)
 

X = np.array(ct.fit_transform(X),  dtype=float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=0.2, random_state=42)