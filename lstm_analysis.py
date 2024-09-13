import pandas as pd
import numpy as np

training_set = pd.read_csv(r"/Users/mariochima/Desktop/my first folder/coding folder/machine learning practice/traffic flow/train.csv", index_col= 0)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None) 

# print(training_set.head())

lstm_data = training_set.iloc[:, 0: 13]
# print(lstm_data.head())

#there are 36 different locations
#each location refers to 1261 timestamps (each time stamp are 15 min intervals)

#each location has 1261 timestamps - you need to group each dataframe with their timestep and location

lstm_data_jan = lstm_data[lstm_data['location'] == 23]
print(lstm_data_jan.head(20))

#traffic values do no align with prev values so I am going to make prev_1 the traffic value, adjust the other columns, and then remove traffic
#clean up data (making each dataframe for each location and then organizing them in a dictionary)
#cont to reshape data and then use pytorch