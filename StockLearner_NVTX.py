#!/usr/bin/env python
# coding: utf-8

# ## Setup and Initialization
# Let's start by setting up the environment. Specifically, we need to ensure we have a CUDA capable device. If we don't, we can fall back to CPU. Setting the device value here allows us to forcefully override this value if ever required.

# In[1]:


import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[2]:


import nvtx
import time


# ## Data Loader Generator
# 
# Let's build a class that can extract data from a given CSV file and build TensorDatasets from the data.
# 
# ### Constructor 
# The constructur takes two parameters, the CSV file path and a debug flag. The Debug flag is used across the class to print out helpful debugging data as we process the CSV data.
# 
# ### preprocess_date
# This method centers the transaction date around a specific "Epoch" date. Currently the epoch date of choice is the start of hte year we're currently processing.
# 
# ### Normalize
# The normalize method fits the specified columns with a MinMaxScaler. This reduces all values in that column to a \[0,1\] range.

# In[3]:


import pandas as pd
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.data import (TensorDataset, DataLoader)

class CSVDataSetLoader:
    def __init__(self, csv_file, debug=False):
        self.debug = debug
        
        raw_data = pd.read_csv(csv_file)
        
        # TODO (sahil.ramani): If the file data is reversed, reverse it.
        # Reverse the data frame since we get the data backwards
        self.raw_data = raw_data[::-1].reset_index(drop=True)
        
        # Create a new index column
        self.raw_data.reset_index(level=0, inplace=True)

        if debug:
            print(self.raw_data.head())

        self.start_of_year = datetime.strptime(f'{year}-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        
    def set_debug(self, debug=True):
        self.debug = debug


    @nvtx.annotate("preprocess_date()", color="purple")
    def preprocess_date(self, *drop_columns):
        # Create a new field containing a coalesced "date since data collection start" entry
        # In this case, we're calling the field 'datetime_epoch' where epoch just means start of year for us right now.

        def dt_since_year_start_seconds(dt):
            return (dt-self.start_of_year).total_seconds()

        raw_data['datetime_epoch'] = [dt_since_year_start_seconds(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) for x in raw_data['datetime']]
        raw_data['datetime_epoch'] = [int((x - raw_data['datetime_epoch'][0])/60) for x in raw_data['datetime_epoch']]


        # Let's visualize the data we have now
        if self.debug:
            raw_data.head()
        
    
    @nvtx.annotate("normalize()", color="blue")
    def normalize(self, *columns):
        # normalize 'close' and 'volume'
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        self.normalize_range = scaler.fit(self.raw_data[list(columns)])
        self.raw_data[[x + '_norm' for x in columns]] = scaler.transform(self.raw_data[list(columns)])
        
        if self.debug:
            print(self.normalize_range.data_min_, self.normalize_range.data_max_)
            self.raw_data.head()
            
            
    @nvtx.annotate("drop_columns()", color="green")
    def drop_columns(self, *columns):
        
        # Now we just drop the other fields in the dataset because we technically don't need them.
        self.raw_data = self.raw_data.drop(columns=list(columns))

        if self.debug:
            print(self.raw_data.head())

    @nvtx.annotate("drop_columns_except()", color="yellow")
    def drop_columns_except(self, *columns):

        # Now we just drop the other fields in the dataset because we technically don't need them.
        self.raw_data = self.raw_data.drop(columns=list(set(self.raw_data.columns)-set(columns)))
        
        if self.debug:
            print(self.raw_data.head())
        
        
    @nvtx.annotate("create_datasets()", color="red")
    def create_datasets(self, train_percentage=0.5, val_percentage=0.3, test_percentage=0.2):
        if train_percentage + val_percentage + test_percentage != 1.0:
            print("Invalid percentages. Make sure the split adds up to 1.")
        
        # Split our raw data into training-validation-test sets

        total_entries = self.raw_data.shape[0]

        split_idx = int(total_entries*train_percentage)
        self.train_data, remaining_data = self.raw_data[:split_idx], self.raw_data[split_idx:]

        test_idx = int(remaining_data.shape[0]*val_percentage)
        self.val_data, self.test_data = remaining_data[:test_idx], remaining_data[test_idx:]
        
        if self.debug:
            ## print out the shapes of your resultant feature data
            print("\t\t\tFeature Shapes:")
            print("Train set: \t\t{}".format(self.train_data.shape), 
                  "\nValidation set: \t{}".format(self.val_data.shape),
                  "\nTest set: \t\t{}".format(self.test_data.shape))

    @nvtx.annotate("create_dataloaders()", color="purple")
    def create_dataloaders(self, batch_size, seq_length, train_percentage=0.5, val_percentage=0.3, test_percentage=0.2):
        
        self.create_datasets(train_percentage, val_percentage, test_percentage)
        
        # Data Preprocessing
        train_x = [self.train_data[i: i+seq_length]['close_norm'].to_numpy() for i in range(0, len(self.train_data) - seq_length - 1)]
        train_y = [self.train_data[i: i+seq_length]['close_norm'].to_numpy() for i in range(1, len(self.train_data) - seq_length)]

        for i in range(len(train_x)):
            train_x[i].resize(seq_length, 1)
        for i in range(len(train_y)):
            train_y[i].resize(seq_length, 1)    

        self.time_steps_train = [self.train_data[i: i+seq_length]['index'].to_numpy(dtype=np.float32) for i in range(1, len(self.train_data) - seq_length)]

        val_x = [self.val_data[i: i+seq_length]['close_norm'].to_numpy() for i in range(0, len(self.val_data) - seq_length - 1)]
        val_y = [self.val_data[i: i+seq_length]['close_norm'].to_numpy() for i in range(1, len(self.val_data) - seq_length)]

        for i in range(len(val_x)):
            val_x[i].resize(seq_length, 1)
        for i in range(len(val_y)):
            val_y[i].resize(seq_length, 1)    

        self.time_steps_val = [self.val_data[i: i+seq_length]['index'].to_numpy() for i in range(1, len(self.val_data) - seq_length)]

        test_x = [self.test_data[i: i+seq_length]['close_norm'].to_numpy() for i in range(0, len(self.test_data) - seq_length - 1)]
        test_y = [self.test_data[i: i+seq_length]['close_norm'].to_numpy() for i in range(1, len(self.test_data) - seq_length)]

        for i in range(len(test_x)):
            test_x[i].resize(seq_length, 1)
        for i in range(len(test_y)):
            test_y[i].resize(seq_length, 1)    

        self.time_steps_test = [self.test_data[i: i+seq_length]['index'].to_numpy() for i in range(1, len(self.test_data) - seq_length)]

        # create Tensor datasets
        train_dataset = TensorDataset(torch.as_tensor(train_x).float().to(device), torch.as_tensor(train_y).float().to(device))
        valid_dataset = TensorDataset(torch.as_tensor(val_x).float().to(device), torch.as_tensor(val_y).float().to(device))
        test_dataset = TensorDataset(torch.as_tensor(test_x).float().to(device), torch.as_tensor(test_y).float().to(device))

        # dataloaders
        # make sure the SHUFFLE your training data
        self.training = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.validation = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
        self.test = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
        
        if self.debug:
            print(self.training, self.validation, self.test)


# ## Training Session
# Creates a training session class that can take a model, CSVDataSetLoader, loss function and an optimizer type, and provide simplified methods to train, validate and test models. Eventually this class will serve as the foundation to checkpoint and save out models as well.

# In[4]:


class TrainingSession:
    def __init__(self, model, loader, criterion, optimizer_type, lr):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr)
        else:
            print("Invalid optimizer type")
        self.lr = lr
        
    @nvtx.annotate("train_epoch()", color="purple")
    def train_epoch(self, epoch, print_iter=1000):
        
        self.model.train()
        losses = []
        
        # Initialize Hidden State
        hidden = self.model.init_hidden()
        range_end = len(self.loader.training)
        index=0
        for input_t, target_t in self.loader.training:

            index+=1
            # Zero out gradients.
            self.optimizer.zero_grad()

            # Run this tensor through the model
            prediction, hidden = self.model(input_t, hidden)

            # Detach from previous state so you're not tracking gradients
            if isinstance(hidden, tuple):
                hidden = tuple([x.data.type(torch.FloatTensor).to(device) for x in hidden])
            else:
                hidden = hidden.data.type(torch.FloatTensor).to(device)

            # Calculate Loss
            loss = self.criterion(prediction.reshape(target_t.shape), target_t)

            losses.append(loss.item())
            
            # Backprop
            loss.backward()
            self.optimizer.step()

            if index % print_iter == 0:
                print('Epoch : {}, Batch : {:2.2%}, Loss: {}'.format(epoch+1, (index/range_end), loss.item()))

        print('-------')
        print('Average Training Loss : {}'.format(np.mean(losses)))
        print('-------')
        
    @nvtx.annotate("train()", color="red")
    def train(self, epochs, print_iter=1000):
        for epoch in range(epochs):
            self.train_epoch(epoch, print_iter)
            
    @nvtx.annotate("validate()", color="yellow")
    def validate(self):
        self.model.eval()
        losses = []
        for input_t, target_t in self.loader.validation:
            prediction, _ = self.model(input_t.to(device), None)
            loss = self.criterion(prediction.reshape(target_t.shape), target_t)
            losses.append(loss.item())
            
        print('-------')
        print('Average Validation Loss : {}'.format(np.mean(losses)))
        print('-------')
            
    @nvtx.annotate("test()", color="green")
    def test(self):
        self.model.eval()
        losses = []
        for input_t, target_t in self.loader.test:
            prediction, _ = self.model(input_t.to(device), None)
            loss = self.criterion(prediction.reshape(target_t.shape), target_t)
            losses.append(loss.item())
                    
        print('-------')
        print('Average Test Loss : {}'.format(np.mean(losses)))
        print('-------')


# In[5]:


# Alright, let's build our RNN now
class StockRNN(nn.Module):
    @nvtx.annotate("StockRNN::Init()", color="red")
    def __init__(self, input_size, output_size, hidden_dim, layer_count, batch_size, dropout=0.25):
        super(StockRNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.batch_size = batch_size
        
        self.rnn = nn.RNN(input_size, hidden_dim, layer_count, dropout=dropout, nonlinearity='relu', batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    @nvtx.annotate("StockRNN::forward()", color="yellow")
    def forward(self, data, hidden):
        out_rnn, out_hidden = self.rnn(data, hidden)
        out_rnn = out_rnn.view(-1, self.hidden_dim)
        out_data = self.fc(out_rnn)
        
        return out_data, out_hidden
    
    @nvtx.annotate("StockRNN::init_hidden()", color="green")
    def init_hidden(self):
        return torch.zeros(self.layer_count, self.batch_size, self.hidden_dim).to(device)


# In[6]:


# Additionally, let's build ourselves a GRU network
class StockGRU(nn.Module):
    @nvtx.annotate("StockGRU::Init()", color="red")
    def __init__(self, input_size, output_size, hidden_dim, layer_count, batch_size, dropout=0.25):
        super(StockGRU, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.batch_size = batch_size
        self.dropout_prob = dropout
        
        self.gru = nn.GRU(input_size, hidden_dim, layer_count, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
        
    @nvtx.annotate("StockGRU::forward()", color="yellow")
    def forward(self, data, hidden):
        out_gru, out_hidden = self.gru(data, hidden)
        out_data = self.fc(self.relu(out_gru))
        return out_data, out_hidden
    
    @nvtx.annotate("StockGRU::init_hidden()", color="green")
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.layer_count, self.batch_size, self.hidden_dim).zero_().to(device)
        return hidden


# In[7]:


# and an LSTM network
class StockLSTM(nn.Module):
    @nvtx.annotate("StockLSTM::Init()", color="red")
    def __init__(self, input_size, output_size, hidden_dim, layer_count, batch_size, dropout=0.25):
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.batch_size = batch_size
        self.dropout_prob = dropout
        
        self.lstm = nn.LSTM(input_size, hidden_dim, layer_count, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    @nvtx.annotate("StockLSTM::forward()", color="yellow")
    def forward(self, data, hidden):
        out_lstm, out_hidden = self.lstm(data, hidden)
        out_data = self.fc(out_lstm)
        return out_data, out_hidden
        
    
    @nvtx.annotate("StockLSTM::init_hidden()", color="green")
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.layer_count, self.batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.layer_count, self.batch_size, self.hidden_dim).zero_().to(device))
        return hidden


# In[8]:


# Parameters to feed into dataset loading
data_dir = './data/'
raw_input_dir = 'raw/'
symbol = 'TSLA'
interval = '1min'
year = '2020'

filename = f'{symbol}_{year}_{interval}.csv'
full_file_path = data_dir + raw_input_dir + filename

csv_loader = CSVDataSetLoader(full_file_path, debug=True)
csv_loader.drop_columns_except('volume', 'close', 'index')
csv_loader.normalize('volume', 'close')
csv_loader.create_dataloaders(batch_size=1, seq_length=200)


# In[9]:


# RNN Setup/Hyperparameters
input_size=1 
output_size=1
hidden_dim=128
layer_count=2
batch_size=1

# Create stock learner models
rnn_model = StockRNN(input_size, output_size, hidden_dim, layer_count, batch_size)
rnn_model.to(device)
print(rnn_model)

gru_model = StockGRU(input_size, output_size, hidden_dim, layer_count, batch_size)
gru_model.to(device)
print(gru_model)

lstm_model = StockLSTM(input_size, output_size, hidden_dim, layer_count, batch_size)
lstm_model.to(device)
print(lstm_model)


# In[10]:


rnn_session = TrainingSession(rnn_model, csv_loader, nn.MSELoss(), 'adam', lr=1e-5)


# In[11]:


with nvtx.annotate("RNN Training", color="green"):
    rnn_session.train(epochs=2)


# In[ ]:


lstm_session = TrainingSession(lstm_model, csv_loader, nn.MSELoss(), 'adam', lr=1e-5)
with nvtx.annotate("LSTM Training", color="blue"):
    lstm_session.train(epochs=2)


# In[ ]:


gru_session = TrainingSession(gru_model, csv_loader, nn.MSELoss(), 'adam', lr=1e-5)
with nvtx.annotate("GRU Training", color="red"):
    gru_session.train(epochs=2)


# In[ ]:


gru_session.validate()


# In[ ]:


gru_session.test()


# In[ ]:




