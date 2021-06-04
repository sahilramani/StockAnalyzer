import numpy as np
import torch.nn as nn
import torch

class StockGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, fanout_size, output_size, dropout_prob, device):
        super(StockGRU, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fanout_size = fanout_size
        self.output_size = output_size
        self.device = device
        
        # If we only have one layer, disable dropout since it won't work.
        if num_layers == 1:
            dropout_prob = 0

        self.dropout_prob = dropout_prob
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.fc_1 =  nn.Linear(hidden_size, fanout_size)
        self.fc = nn.Linear(fanout_size, output_size)
        self.relu = nn.ReLU()
        
    
    def forward(self,x):
        batch_size = x.size(0)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # Propagate input through GRU
        output, hn = self.gru(x, h)
        # Prepare LSTM output for fully conncected layer by flattening it.
        out = output.contiguous().view(-1, self.hidden_size)
        out = self.relu(out)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        
        # Reshape the output back wrt batch size
        out = out.view(batch_size, -1, self.output_size)
        
        # Take the last batch of outputs
        out = out[:, -1]
        return out
    
    
    def save(self, file, train_loss, val_loss, epoch, lr):
        torch.save({
            'num_layers'  : self.num_layers,
            'input_size'  : self.input_size,
            'hidden_size' : self.hidden_size,
            'fanout_size' : self.fanout_size,
            'output_size' : self.device,
            'train_loss'  : train_loss,
            'val_loss'    : val_loss,
            'epoch'       : epoch,
            'lr'          : lr,
            'data'        : self.state_dict()
        }, file);
    
    
    @classmethod
    def from_checkpoint(cls, file):
        ckpt = torch.load(file)
        model = cls(ckpt['num_layers'], ckpt['input_size'], ckpt['hidden_size'], ckpt['fanout_size'], ckpt['output_size'], ckpt['dropout_prob'], ckpt['device'])
        model.load_state_dict(ckpt['data'])
        model.to(ckpt['device'])
        return model