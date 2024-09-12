import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

device = torch.device('mps')

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
        x = x.to(device)
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
net.to(device)

# train the neural network
def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    model.train()

    for epoch in range(10):
        
        batch_loss = 0

        for batch, (X, y) in enumerate(data_loader):
            
            X = X.to(device)
            y = y.to(device)

            # compute prediction and loss 
            y_pred = model(X)
            error = loss_fn(y_pred,y)

            batch_loss += error
        
            # backpropagation
            error.backward()
            optimizer.step()
            optimizer.zero_grad()
                         
        # show testing acuracy in each iteration of the training function
        train_accuracy = eval(trainloader, model, loss_fn)
        testing_accuracy = eval(testloader, model, loss_fn)
        avg_batch_loss = batch_loss / len(data_loader)

        # print the loop, train loss, train acc %, test loss, test acc %
        print(f"{epoch + 1:<4}\t{avg_batch_loss:.4f}\t{100 * train_accuracy[0]:.3f}\t{testing_accuracy[1]:3f}\t{100 * testing_accuracy[0]:.3f}")
         
    # save the model after training is complete
    torch.save(model.state_dict(), 'model/model.pth')

# evaluate neural network accuracy 
def eval(data_loader, model, loss_fn):
    model.eval()
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            pred_y = model(X)
            test_loss += loss_fn(pred_y, y).item()
            accuracy += torch.sum(pred_y.argmax(1) == y).item()
    
    avg_test_loss = test_loss / len(data_loader)
    accuracy /= len(data_loader.dataset)

    return accuracy, avg_test_loss

def test(image_path):
    with open('model/model.pth', 'rb') as f:
        net.load_state_dict(torch.load(f))

    net.eval()
    img = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    output = torch.max(net(img_tensor), 1)

    return output.item()


if __name__ == "__main__":
    # print the loop, train loss, train acc %, test loss, test acc %
    if sys.argv[1] == "train":
        print(f"Loop,\tTrain Loss,\tTrain Acc%,\tTest Loss,\tTest Acc%")
        train(trainloader, net, loss, optim)
    elif sys.argv[1] == "test" and sys.argv[2] == './images/png':
        output = test('images/png')
        print("prediction result: ", output)
    else:
        print("please enter the correct function name")
        
