# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 2
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import json, argparse
import torch.optim as optim
import torch as torch
import numpy as np
from data_parse import Data
from nn_gen import Net
import matplotlib.pyplot as plt

# Function finds the number of correct predictions given a list of training and test data
def test(x_list,y_list):
    i = 0
    correct = 0
    for x in x_list:
        if (np.argmax(x.detach().numpy()) == np.argmax(y_list[i].numpy())):
            correct += 1
        i += 1

    # Calc Average
    correct = correct/len(x_list) * 100

    return correct

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Multi-Label Classifier in PyTorc")
    parser.add_argument("params",metavar="params/param_file_name.json",type=str)
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

    # Create our network and dataset
    model = Net()
    print("Importing data.....")
    data = Data("even_mnist.csv",3000)
    print("Done")

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(),lr=hyper["learning rate"],momentum=0.9)
    loss_func = torch.nn.BCELoss(reduction="mean")
        
    # Number of training epochs
    num_epochs = hyper["num epochs"]

    # lists to track preformance of network
    obj_vals= []
    cross_vals= []

    # Training loop
    for epoch in range(1, num_epochs + 1):

        # Clear our gradient buffer
        optimizer.zero_grad()

        # Clear gradients
        model.zero_grad()

        # feed our inputs through the net
        output = model(data.x_train)

        # Calculate our loss
        loss = loss_func(output,data.y_train)

        # Backpropagate our loss
        loss.backward()

        # Graph our progress
        obj_vals.append(loss)
        test_val= model.test(data, loss_func, epoch)
        cross_vals.append(test_val)

        optimizer.step()

        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch + 1) % hyper["display epochs"]):
                print('Epoch [{}/{}]'.format(epoch, num_epochs) +\
                    '\tTraining Loss: {:.4f}'.format(loss) +\
                    '\tTest Loss: {:.4f}'.format(test_val) +\
                    "\tPercent Correct: {:.2f}".format(test(output,data.y_train)))

    # Low verbosity final report
    if args.v:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
        print("Final percent correct: {:.2f}".format(test(output,data.y_train)))

    # Plot Results
    plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    plt.legend()
    plt.show()


   