import json, torch
import torch.optim as optim
from nn_gen import Net
from data_gen import Data
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_style("darkgrid")

if __name__ == '__main__':

    # Command line arguments
    arg = sys.argv
    param_file = arg[1]

    # Hyperparameters from json file
    with open(param_file) as paramfile:
        param = json.load(paramfile)

    # Construct a model and dataset
    model= Net()
    data= Data(param['data_file'], int(param['n_test_data']))

    # Define an optimizer and the loss function
    optimizer = optim.SGD(model.parameters(), lr=param['learning_rate'])
    loss= torch.nn.BCELoss()

    obj_vals= []
    cross_vals= []
    num_epochs= int(param['num_epochs'])

    # Training loop
    for epoch in range(1, num_epochs + 1):

        train_val= model.backprop(data, loss, epoch, optimizer)
        obj_vals.append(train_val)

        test_val= model.test(data, loss, epoch)
        cross_vals.append(test_val)

        # report progress in output stream
        if not ((epoch + 1) % param['display_epochs']):
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                  '\tTraining Loss: {:.4f}'.format(train_val)+\
                  '\tTest Loss: {:.4f}'.format(test_val))

    # final report
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    plt.legend()
    plt.savefig('results/Loss.pdf')
    plt.close()


