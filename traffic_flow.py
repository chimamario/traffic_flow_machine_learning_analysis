import pandas as pd
import numpy as np
import networkx as nx

#notes about csv files so far

#training and test data sets

#there are 36 different locations
#each location refers to 1261 timestamps (each time stamp are 15 min intervals)
#ex - one row refers to a location, at a specific timestamp. it has the traffic rate at that point as well as 

training_set = pd.read_csv(r"/Users/mariochima/Desktop/my first folder/coding folder/machine learning practice/traffic flow/train.csv")


pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None) 
# print(training_set.head())
# print(training_set.columns)

#testing
#testing 2


# # graph_features.to_csv('graph_features.csv', index=False)
