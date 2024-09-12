import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

#device = torch.device('mps')

# load the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load data, save it to data folder, and load data into train and test loaders
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)


# build the neural network without using nn.Conv2d and MaxPool2d
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3072, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    # forward pass
    def forward(self, x):
        #x.to(device)
        x = x.view(-1, 3072)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.out(x)
        return x

# initialize the neural network and define the loss function and optimizer
net = NeuralNetwork()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)
#net.to(device)

# train the neural network
def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()

    for epoch in range(10):
        
        for batch, (X, y) in enumerate(data_loader):
            
            #X = X.to(device)
            #y = y.to(device)

            # compute prediction and loss 
            y_pred = model(X)
            error = loss_fn(y_pred,y)
        
            # backpropagation
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
                         
        # show testing acuracy in each iteration of the training function
        train_accuracy = eval(trainloader, model, loss_fn)
        testing_accuracy = eval(testloader, model, loss_fn)
        # print the loop, train loss, train acc %, test loss, test acc %
        print("{:4d} | {:.6f} | {:.6f} | {:.6f}".format(epoch + 1, loss, train_accuracy, testing_accuracy))
         
        
    # save the model after training is complete
    torch.save(model.state_dict(), 'model/model.pth')

# test the neural network
def eval(data_loader, model, loss_fn):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            #X.to(device)
            #y.to(device)

            pred_y = model(X)
            test_loss += loss_fn(pred_y, y).item()
            correct += torch.sum(pred_y.argmax(1) == y).item()
    
    test_loss /= len(data_loader)
    accuracy /= len(data_loader.dataset)

    return accuracy

if __name__ == "__main__":
    # print the loop, train loss, train acc %, test loss, test acc %
    print("Loop, ", "Train Loss, ", "Train Acc %, ", "Test Loss, ", "Test Acc%")
    train(trainloader, net, loss, optim)
