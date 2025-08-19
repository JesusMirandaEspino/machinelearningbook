# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 07:08:13 2025



@author: jesus
"""

import numpy as np # aplicacion de matematicas
import matplotlib as plt # mostrar datos
import pandas as pd # manipular datos


dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Data.csv')

X = dataset.iloc[ :,:-1 ].values
y = dataset.iloc[ :,3 ].values