# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 2
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import torch as torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    ## Arciecture:
    ## Two fully connected layers fc1 and fc2

    def __init__(self):
        super(Net, self).__init__()

        # Create the layers of the neural net
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3)
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=18,kernel_size=3)
        self.pool2 = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(18*2*2, 18*2*2*10)
        self.fc2 = nn.Linear(18*2*2*10, 100)
        self.fc3 = nn.Linear(100, 5)


    # Feedforward function
    def forward(self,x):

        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.reshape(x,(x.size()[0],18*2*2))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        y = torch.sigmoid(self.fc3(x))

        return y      

    # Reset training weights
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

        # Test function. Avoids calculation of gradients.
    def test(self, data, loss_func, epoch):
        self.eval()
        with torch.no_grad():
            inputs= data.x_test
            targets= data.y_test
            outputs= self(inputs)
            cross_val= loss_func(self.forward(inputs).reshape(-1), targets)
        return cross_val.item()

