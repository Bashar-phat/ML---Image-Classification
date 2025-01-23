import numpy as np
import torch
import torch.nn as nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Base channel size set to 56
        self.n = 56  
        # Kernel size and padding
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2  # => 2 for kernel_size=5

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.n,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.n,
            out_channels=2*self.n,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        self.conv3 = nn.Conv2d(
            in_channels=2*self.n,
            out_channels=4*self.n,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        self.conv4 = nn.Conv2d(
            in_channels=4*self.n,
            out_channels=8*self.n,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # After 4 conv+pool layers on [3, 448, 224]:
        #   448->224->112->56->28 in height
        #   224->112->56->28->14 in width
        # Final channels => 8*self.n = 8*56 = 448
        # Flattened => 448 * 28 * 14 = 175616
        self.fc1 = nn.Linear(8*self.n * 28 * 14, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.relu(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)
        
        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool(out)
        
        out = out.reshape(out.size(0), -1)  # Flatten
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out



class CNNChannel(nn.Module):
    def __init__(self):
        super(CNNChannel, self).__init__()
        # n=32 for the initial base channel size
        self.n = 32

        # We'll use kernel_size=4, padding=2
        self.conv1 = nn.Conv2d(in_channels=6,   # left+right shoes => 6 channels
                               out_channels=self.n, 
                               kernel_size=4,
                               padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.n,
                               out_channels=2*self.n,
                               kernel_size=4,
                               padding=2)
        self.conv3 = nn.Conv2d(in_channels=2*self.n,
                               out_channels=4*self.n,
                               kernel_size=4,
                               padding=2)
        self.conv4 = nn.Conv2d(in_channels=4*self.n,
                               out_channels=8*self.n,  # => 256
                               kernel_size=4,
                               padding=2)

        # Max-pooling after each convolution
        self.pool = nn.MaxPool2d(2, 2)

        # After 4 pooling steps on (224×224):
        #  224 → 112 → 56 → 28 → 14 in each spatial dimension
        # Final channels = 8*self.n = 8*32 = 256
        # Flattened => 256 * 14 * 14 = 50176
        self.fc1 = nn.Linear(50176, 100)
        self.fc2 = nn.Linear(100, 2)

        self.relu = nn.ReLU()

    def forward(self, inp):
        # Split top (left shoe) and bottom (right shoe) each of shape [3,224,224]
        left_shoe = inp[:, :, :224, :]   # => (batch, 3, 224, 224)
        right_shoe = inp[:, :, 224:, :]  # => (batch, 3, 224, 224)

        # Concatenate along channel dimension => shape [batch, 6, 224, 224]
        out = torch.cat((left_shoe, right_shoe), dim=1)

        out = self.pool(self.relu(self.conv1(out)))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        out = self.pool(self.relu(self.conv4(out)))

        # Flatten => [batch_size, 50176]
        out = out.reshape(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out



