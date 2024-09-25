import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os


device = torch.device('mps')

# generate make dir for the model to save train weights and biases 
os.makedirs('model', exist_ok=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])


# load data, save it to data folder, and load data into train and test loaders
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        '''In the first CONV layer, the filter size should be 5*5, the stride should be 1, and the total number of filters should be 32.'''
        self.conv1 = nn.Conv(3, 32, 5, stride=1, padding=2)
        # lets have 3 layers of convolutions with relu activation functions
        # in between the convolutions, we will have maxpooling layers
        # batch normalization is also added
        # after the 5th conv, 2 fully connected layers, and the output layer
        #self.conv2 = nn.Conv()

    # forward pass
    def forward(self, x):
        
        return x

net = CNN()
loss = nn.CrossEntropyLoss()
optim = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
net.to(device)


# train the neural network
def train(epochs, data_loader, model, loss_fn, optimizer):
    model.train()

    for epoch in range(epochs):
        batch_loss = 0

        for (X, target) in enumerate(data_loader):      
            X = X.to(device)
            y = target.to(device)

            # compute prediction and loss 
            y_pred = model(X)
            error = loss_fn(y_pred, target)

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
        net.load_state_dict(torch.load(f, weights_only=True))

    net.eval()
    img = Image.open(image_path)
    img_resize = img.resize((32, 32))
    img_tensor = transforms.ToTensor()(img_resize).unsqueeze(0).to(device)
    output = torch.max(net(img_tensor), 1)

    return output


if __name__ == "__main__":
    # print the loop, train loss, train acc %, test loss, test acc %
    if sys.argv[1] == "train":
        print(f"Loop,\tTrain Loss,\tTrain Acc%,\tTest Loss,\tTest Acc%")
        train(30, trainloader, net, loss, optim)
    elif sys.argv[1] == "test" or sys.argv[1] == "predict":
        output = test(sys.argv[2])
        print("prediction result: ", classes[output.indices.item()])
    else:
        print("please enter the correct function name")