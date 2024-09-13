import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


training_set = pd.read_csv(r"/Users/mariochima/Desktop/my first folder/coding folder/machine learning practice/traffic flow/train.csv", index_col= 0)
test_set = pd.read_csv(r"/Users/mariochima/Desktop/my first folder/coding folder/machine learning practice/traffic flow/test.csv", index_col=0)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None) 

#only focusing on the location, timestep, traffic and prev traffic data
lstm_data = training_set.iloc[:, 0: 13]
lstm_test_data = test_set.iloc[:, 0: 13]

#the traffic rates make no sense so we will be using the prev_1 as traffic rate
lstm_data.columns = ['timestep', 'location', 'traffic_old', 'traffic','prev_1', 'prev_2', 'prev_3',
       'prev_4', 'prev_5', 'prev_6', 'prev_7', 'prev_8', 'prev_9']

lstm_test_data.columns = ['timestep', 'location', 'traffic_old', 'traffic','prev_1', 'prev_2', 'prev_3',
       'prev_4', 'prev_5', 'prev_6', 'prev_7', 'prev_8', 'prev_9']

lstm_data = lstm_data.drop('traffic_old', axis = 1)
lstm_test_data = lstm_test_data.drop('traffic_old', axis = 1)

#there are 36 different locations
#each location refers to 1261 timestamps (each time stamp are 15 min intervals)

#each location has 1261 timestamps - you need to group each dataframe with their timestep and location


#creating a train and test set in tensor form
lstm_data_jan = lstm_data[lstm_data['location'] == 23]
lstm_data_jan.reset_index(drop=True, inplace=True)

lstm_test_jan = lstm_test_data[lstm_test_data['location'] == 23]
lstm_test_jan.reset_index(drop=True, inplace=True)

X_train = lstm_data_jan.iloc[:,3:12]
y_train = lstm_data_jan.iloc[:, 2]

X_test = lstm_test_jan.iloc[:,3:12]
y_test = lstm_test_jan.iloc[:, 2]


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) #(1261, 9) (1261,)

#convert to numpy

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) #(1261, 9) (1261,)

#reshaping training data because pytorch needs an extra dimension for the tensor tings

lookback = 9

X_train = X_train.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1,1))

X_test = X_test.reshape((-1, lookback, 1))
y_test = y_test.reshape((-1,1))

#convert arrays to tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()

X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#creating datasets class for pytorch

class TslmDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset =  TslmDataset(X_train, y_train)
test_dataset = TslmDataset(X_test, y_test)

#things to look into - datsets and dataloaders

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle= False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle= False)






#traffic values do no align with prev values so I am going to make prev_1 the traffic value, adjust the other columns, and then remove traffic
#clean up data (making each dataframe for each location and then organizing them in a dictionary)
#cont to reshape data and then use pytorch



# lstm_np = lstm_data.to_numpy()

# print(lstm_np)
# range(lstm_data['location'].nunique())
# lstm_location_dict = {} #this dictionary will store all the lstm_data wrt each location

# for i in range(3) :
#     data = lstm_data[lstm_data['location'] == i]
#     lstm_location_dict[i] = data
    

# print(lstm_location_dict[1])


