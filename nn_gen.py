# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 2
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    ## Arciecture:
    ## Two fully connected layers fc1 and fc2

    def __init__(self, x_len):
        super(Net, self).__init__()
        self.fc1=nn.Linear(x_len,100)
        self.fc2=nn.Linear(100,5)

        