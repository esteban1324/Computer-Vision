import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Check if Metal (GPU support) is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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
        self.dropout = nn.Dropout(0.2)

    # forward pass
    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.out(x)
        return x

# initialize the neural network and define the loss function and optimizer
net = NeuralNetwork()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
net.to(device)

# train the neural network
def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()

    for epoch in range(10):
        
        for batch, (X, y) in enumerate(data_loader):
        
            # compute prediction and loss 
            y_pred = model(X)
            error = loss_fn(y_pred,y)
        
            # backpropagation
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # print the accuracy
            if batch % 100 == 0:
                loss, current = error.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                     
        # show testing acuracy in each iteration of the training function
        testing_accuracy = eval(testloader, model, loss_fn)
        # print the loop, train loss, train acc %, test loss, test acc %
        print(f"Loop: {epoch + 1}, Train loss: {loss:>7f}, Train Acc: {eval(trainloader, model, loss_fn):>7f}%, Test Acc: {testing_accuracy:>7f}%")
         
        
    # save the model after training is complete
    torch.save(model.state_dict(), 'model/model.pth')

# test the neural network
def eval(data_loader, model, loss_fn):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            y_pred = model(X)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(trainloader, net, loss, optim)
    else:
        print("Invalid command. Please use 'python classifier.py train' to train the model.")
