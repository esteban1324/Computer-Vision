import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import torch.utils
from torch.autograd import grad
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
#from InceptionScore import get_inception_score
torch.autograd.set_detect_anomaly(True)
inception = InceptionScore(splits=10, normalize=True).cuda()
fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()


device = torch.device(
    'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockUp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_unsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        shortcut = self.shortcut(x_unsampled)
        
        # residual path 
        x = self.batch_norm1(self.conv1(x_unsampled))
        x = F.relu(x)
        x = self.batch_norm2(self.conv2(x))

        # residual shortcut to skip layers 
        x += shortcut
        x = F.relu(x)

        return x 

# class Generator Network
class Generator(nn.Module):
    def __init__(self, z_dim=128, n_filters=256, img_channels=3):
        super(Generator, self).__init__()
        self.num_channels = img_channels
        self.dense = nn.Linear(z_dim, 4 * 4 * 256)
        self.resblock1 = ResBlockUp(n_filters, n_filters)
        self.resblock2 = ResBlockUp(n_filters, n_filters)
        self.resblock3 = ResBlockUp(n_filters, n_filters)
        self.batch_norm = nn.BatchNorm2d(n_filters)
        self.ReLU = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(n_filters, img_channels, 3, 1, 1) 

    def forward(self, z):
        x = self.dense(z).view(-1, 256, 4, 4)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.batch_norm(x) 
        x = self.ReLU(x)
        x = self.conv(x)
        x = torch.clamp(x, min=-1, max=1)

        return x
 
# Res-blockdown 
class ResBlockDown(nn.Module):
    def __init__(self, in_features, out_features, downsample=False):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_features)

        if downsample or in_features != out_features:
            self.shortcut = nn.Conv2d(in_features, out_features, kernel_size=1, stride=2 if downsample else 1, padding=0)
        else:
            self.shortcut = nn.Identity() # no downsampling for shortcut

        self.ReLU = nn.ReLU(inplace=True) 

    def forward(self, x):
        res = self.ReLU(x)
        res = self.batch_norm1(self.conv1(res))
        res = self.ReLU(res)
        res = self.batch_norm2(self.conv2(res))
        shortcut = self.shortcut(x)

        return res + shortcut

# class Discriminator Network 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.res_block1 = ResBlockDown(3, 128, downsample=True)
        self.res_block2 = ResBlockDown(128, 128, downsample=True)
        self.res_block3 = ResBlockDown(128, 128, downsample=False)
        self.res_block4 = ResBlockDown(128, 128, downsample=False)

        self.ReLU = nn.ReLU(inplace=True)
        self.dense = nn.Linear(128, 1)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.ReLU(x)
        x = torch.sum(x, dim=[2,3])    
        x = self.dense(x)

        return x

lr = 2e-4
beta1 = 0.0
beta2 = 0.9
n_critic = 5  
lambda_gp = 10 

generator = Generator(z_dim=128).to(device)
discriminator = Discriminator().to(device)
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

def gradient_penalty(discriminator, real_data, fake_data):
    batch_size, channels, height, width = real_data.shape
    alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real_data).to(device)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolated = discriminator(interpolated)

    gradients = grad(
        outputs = d_interpolated,
        inputs = interpolated,  
        grad_outputs = torch.ones_like(d_interpolated, device=device),
        create_graph = True,
        retain_graph = True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gp


# train the GAN model 
def train(epochs, data_loader, generator, discriminator, gen_optimizer, disc_optimizer):
    inception_scores = []
    fids = []
    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(data_loader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            print(f"epoch round: {epoch}, batch progress: {i + 1}/{len(data_loader)}")
            for _ in range(n_critic):
                noise = torch.randn(batch_size, 128).to(device)
                fake_data = generator(noise)

                # discriminator loss: D(real) - D(fake)
                disc_real = discriminator(real_data).mean()
                disc_fake = discriminator(fake_data.clone().detach()).mean()
                gp = gradient_penalty(discriminator, real_data, fake_data)
                disc_loss = disc_fake - disc_real + gp

                # backprop and optimization 
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
            
            
            # train generator 
            noise = torch.randn(batch_size, 128).to(device)
            fake_data = generator(noise)
            gen_loss = - discriminator(fake_data).mean()

            # backprop and optimization
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

        
        print(f"Epoch [{epoch+1}/{epochs}]  Discriminator Loss: {disc_loss.item():.4f}  Generator Loss: {gen_loss.item():.4f}")
        grid_size = int(np.ceil(np.sqrt(batch_size)))
        for j in range(batch_size):
            plt.subplot(grid_size, grid_size, j + 1)
            plt.imshow((fake_data[j].permute(1, 2, 0).cpu().detach().numpy() * 0.5 + 0.5).clip(0, 1))
            plt.axis('off')
        
        img = f"epoch{epoch+1}.png"
        plt.savefig(img)
        inception.update(fake_data)
        mean_score, std_score = inception.compute()
        print(f"Inception Score at epoch {epoch}: Inception Score = {mean_score:.2f}, {std_score:.2f}")
        inception_scores.append((mean_score, std_score))
        

        fid.update(real_data, real=True)
        fid.update(fake_data, real=False)
        fid_score = fid.compute()
        print(f"Epoch {epoch + 1}: FID Score = {fid_score:.4f}")
        fids.append(fid_score)
        plt.close()
    print("inception scores: ", inception_scores)
    print("fid scores: ", fids)
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    if sys.argv[1] == "train":
        print("Training GAN")
        train(10, trainloader, generator, discriminator, gen_optimizer, disc_optimizer)
    else:
        print("please enter the correct function name")
