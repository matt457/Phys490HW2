import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        One convolution layer conv1 (plus relu, max pool, then batch norm)
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and sigmoid.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,1,2)
        self.pool = nn.MaxPool2d(2)
        self.batch = nn.BatchNorm2d(1)
        self.fc1= nn.Linear(36, 100)
        self.fc2= nn.Linear(100, 5)

    # Feedforward function
    def forward(self, x):
        conv1 = self.batch(self.pool(func.relu(self.conv1(x))))
        h1 = conv1.reshape(*conv1.shape[:2],-1) # flatten last two dimensions
        h2 = func.relu(self.fc1(h1))
        y = torch.sigmoid(self.fc2(h2))
        
        return y

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.conv1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    # Backpropagation function
    def backprop(self, data, loss, epoch, optimizer):
        self.train()
        inputs= torch.from_numpy(data.x_train)
        targets= torch.from_numpy(data.y_train)
        outputs= self(inputs)
        obj_val= loss(self.forward(inputs), targets)
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss, epoch):
        self.eval()
        with torch.no_grad():
            inputs= torch.from_numpy(data.x_test)
            targets= torch.from_numpy(data.y_test)
            outputs= self(inputs)
            cross_val= loss(self.forward(inputs), targets)
        return cross_val.item()
