# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:06:11 2025

@author: jesus
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from zlib import crc32

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer


housing = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/housing/housing.csv")
housing_full = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/housing/housing.csv")

print(housing.info())


print(housing.value_counts())


print(housing['ocean_proximity'].value_counts())


print(housing.describe())


housing.hist( bins=50, figsize=(12,8) )


def suffle_and_split_data(data, test_ratio):
    suffled_indices = np.random.permutation(len(data))
    test_set_size = int( len(data) * test_ratio)
    test_indices = suffled_indices[:test_set_size]
    train_indices = suffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



train_set, test_set = suffle_and_split_data(housing, 0.2)
print(len(train_set))
print(len(test_set))


def is_id_test_set( identifier, test_ratio ):
    return crc32( np.int64( identifier )) < test_ratio * 2**32


def split_data_with_id_hash( data, test_ratio, id_column ):
    ids = data[id_column]
    int_test_set = ids.apply( lambda id_: is_id_test_set(id_, test_ratio) )
    return data.loc[~int_test_set], data.loc[int_test_set]



housing_with_id = housing.reset_index()
train_set, test_set = split_data_with_id_hash( housing_with_id, 0.2, "index" )


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"] * 1000
train_set, test_set = split_data_with_id_hash( housing_with_id, 0.2, "id" )




train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



housing['income_cat'] = pd.cut( housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5] )

housing['income_cat'].value_counts().sort_index().plot.bar( rot=0, grid=True )
plt.xlabel("Income category")
plt.ylabel("NUmber of district")
plt.show()


splitter = StratifiedShuffleSplit( n_splits=10, test_size=0.2, random_state=42 )
strat_splits = []
for train_index, text_index, in splitter.split(housing, housing['income_cat']):
    strat_train_n = housing.iloc[train_index]
    strat_test_n = housing.iloc[text_index]
    strat_splits.append([strat_train_n, strat_test_n])
    
    
    
strat_train_set, strat_test_set = strat_splits[0]


strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing['income_cat'])


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot( kind="scatter", x='longitude', y="latitude", grid=True )


housing.plot( kind="scatter", x='longitude', y="latitude", grid=True, alpha=0.2 )
plt.show()


housing.plot( kind="scatter", x='longitude', y="latitude", grid=True, s=housing["population"] / 100, label="Population",
             c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))
plt.show()

housing_ = housing.copy()
housing_.drop("ocean_proximity",  axis=1, inplace=True)
corr_matrix = housing_.corr()
print( corr_matrix["median_house_value"].sort_values(ascending=False) )


attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


housing.plot( kind="scatter", x='median_income', y="median_house_value", grid=True, alpha=0.1 )
plt.show()


housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]



housing_ = housing.copy()
housing_.drop("ocean_proximity",  axis=1, inplace=True)
corr_matrix = housing_.corr()
print( corr_matrix["median_house_value"].sort_values(ascending=False) )


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()



imputer = SimpleImputer( strategy="median" )

housing_num = housing.select_dtypes( include=[np.number] )

imputer.fit(housing_num)


print(imputer.statistics_)
print(housing_num.median().values)


X = imputer.transform(housing_num)

housing_str = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


















