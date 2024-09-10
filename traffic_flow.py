import pandas as pd

#notes about csv files so far

#training and test data sets

#there are 35 different locations
#each location refers to 1261 timestamps (each time stamp are 15 min intervals)
#ex - one row refers to a location, at a specific timestamp. it has the traffic rate at that point as well as 

training_set = pd.read_csv(r"/Users/mariochima/Desktop/my first folder/coding folder/machine learning practice/traffic flow/train.csv")

print(training_set.head())
print(training_set['location'].unique())

#testing