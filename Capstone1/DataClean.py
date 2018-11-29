#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

templist = []
templist1 = []

#variables for file path
filepath1 = "C:\\Users\\Stella\\Miniconda3\\envs\\py3k\\Tools\\Capstone1\\CRtest.csv"
filepath2 = "C:\\Users\\Stella\\Miniconda3\\envs\\py3k\\Tools\\Capstone1\\CRtrain.csv"

#read training .csv and put into pandas dataframe trainDF
trainDF = pd.read_csv(filepath2)
#explore dataframe
#print(trainDF.head())   
#print(trainDF.tail())
#print(trainDF.shape)
#print(trainDF.info())
#print(trainDF.describe())
#print(trainDF.dtypes)

#find object types (columns: Id, idhogar, dependence, edjefe, and edjefa) all other types are of float64 and int64
objctType = trainDF.dtypes == object
#print(trainDF.loc[:,objctType])

#find columns containing null values
null_columns = trainDF.columns[trainDF.isnull().any()]
#print out which columns have null values and how many rows contain null values    
#print(trainDF[null_columns].isnull().sum())

#compare columns r4t3 = Total persons in household and tamviv = number of persons living in household
#print(trainDF['r4t3'].equals(trainDF['tamviv']))
#compare columns r4t3 = Total persons in household and hogar_total = # of total individuals in household
#print(trainDF['r4t3'].equals(trainDF['hogar_total']))
#compare columns tamviv = number of persons living in household and hogar_total = # of total individuals in household
#print(trainDF['tamviv'].equals(trainDF['hogar_total']))
#compare colums hhsize = houlsehold size and tamhog = size of household
#print(trainDF['hhsize'].equals(trainDF['tamhog']))
#compare columns agesq = 'Age squared' and SQBage = 'age squared'
#print(trainDF['agesq'].equals(trainDF['SQBage']))

#drop duplicate columns and columns with excessive amounts of null values v18q1 (number of tablets household owns) and rez_esc (number of years behind in school)
trainDF = trainDF.drop(['agesq', 'tamhog', 'v18q1', 'rez_esc', 'v2a1'], axis=1)
#print(trainDF.shape)

#drop rows with missing values in meaneduc (average years of education for adults 18+), and SQBmeaned (square of meanduc row).  Dropping one will drop both
trainDF = trainDF.dropna(subset = ['meaneduc'])

#drop duplicate rows print shape to see frow counts, no duplicate rows existed so none were dropped
trainDF = trainDF.drop_duplicates()
#print(trainDF.shape)

#save column names as list
columns = list(trainDF.columns.values)
#print(columns)

#Test again to find columns with missing value to make sure there aren't any
null_columns = trainDF.columns[trainDF.isnull().any()]
#print out which columns have null values and how many rows contain null values    
#print(trainDF[null_columns].isnull().sum())
#print(trainDF.shape)

#find only numeric columns and calculate standard deviation for each
numeric_cols = trainDF.select_dtypes(include=[np.number]).columns
colsStdDev = trainDF[numeric_cols].std()

#find any columns with a standard deviation of 0 and drop them from the numeric_cols list
for index, value in colsStdDev.iteritems():
	if value == 0.0:
		templist.append(index)


#drop columns from trainDF that have a standard deviation of 0 (all row values for column are same value)
trainDF = trainDF.drop(templist, axis=1)
#find numeric columns again
numeric_cols = trainDF.select_dtypes(include=[np.number]).columns
#print(len(numeric_cols))

#find columns that have more than 2 values (most columns contain a 1 value for yes and a 0 value for no)
for column in numeric_cols:
	if (trainDF[column].value_counts(dropna=False).count() > 2):
		templist1.append(column)

#apply zscore to remaining columns 
zscoreNumericCols = trainDF[templist1].transform(zscore)
outliers = ((zscoreNumericCols[templist1] < -3) | (zscoreNumericCols[templist1] > 3))

#run a describe for columns
#print(zscoreNumericCols.describe())