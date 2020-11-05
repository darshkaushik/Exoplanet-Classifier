# torch imports
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Linear, MaxPool1d, ReLU, Sigmoid, BatchNorm1d, Dropout

class Classifier(nn.Module):
    """
    Performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train the classifier in PyTorch, use BCEWithLogitsLoss with 'pos_weights argument'.
    
    """

    def __init__(self):
        super(Classifier, self).__init__()
        
        # Layers
        self.conv1 = Conv1d(in_channels=1, out_channels=8, kernel_size=11)
        self.conv2 = Conv1d(8,16,11)
        self.conv3 = Conv1d(16,32,11)
        self.conv4 = Conv1d(32,64,11)
        
        self.pool = MaxPool1d(4, stride=2)
        
        self.fc1 = Linear(12032, 5000)
        self.fc2 = Linear(5000, 1000)
        self.fc3 = Linear(1000, 100)
        self.fc4 = Linear(100, 1)
        
        self.drop = Dropout(0.3)
        self.Drop = Dropout(0.5)
        
        self.relu = ReLU()
        self.sig = Sigmoid()
        
        self.flat = nn.Flatten()
        
        self.bn1 = BatchNorm1d(8)
        self.bn2 = BatchNorm1d(16)
        self.bn3 = BatchNorm1d(32)
        self.bn4 = BatchNorm1d(64)
    
    def forward(self, b):
        """
        Perform a forward pass of our model on input features, x.
        :param b: A batch of input features of size (batch_size, channels, no. of timestamps)
        :return: A single, sigmoid-activated value as output
        """
        #convolutions
        b= self.bn1(self.pool(self.relu(self.conv1(b))))
        b= self.bn2(self.pool(self.relu(self.conv2(b))))
        b= self.bn3(self.pool(self.relu(self.conv3(b))))
        b= self.bn4(self.pool(self.relu(self.conv4(b))))
        
        #linear
        b= self.Drop(self.flat(b))
        b= self.drop(self.relu(self.fc1(b)))
        b= self.drop(self.relu(self.fc2(b)))
        b= self.drop(self.relu(self.fc3(b)))
        b= self.sig(self.fc4(b))
        
        return b


