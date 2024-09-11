# create a NN that classifies images of CIFAR-10 dataset using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np


# load the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

    
dataiter = iter(trainloader)
images, labels = next(dataiter)

print("tensor shape: ", images.shape)
print("label: ", labels)
print("tensor data: ", images[0])
    
    

# build the neural network without using nn.Conv2d and MaxPool2d
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # flatten the input tensor so it becomes a 1D tensor
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
                
        



