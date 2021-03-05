import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary


class AutoEncoder(nn.Module):

    def __init__(self, kernel_size=(1, 5)):
        super(AutoEncoder, self).__init__()
        # input size = (batch_size, 1, 1, 1024)

        # encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, padding=(0, 2))  # (64, 1, 1024)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel_size, padding=(0, 2))  # (32, 1, 512)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=kernel_size, padding=(0, 2))  # (4, 1, 256)
        self.conv4 = nn.Conv2d(4, 2, kernel_size=kernel_size, padding=(0, 2))  # (2, 1, 128)

        # decoder
        self.inv_conv1 = nn.ConvTranspose2d(2, 2, kernel_size=kernel_size, padding=(0, 2))  # (2, 1, 64)
        self.up1 = nn.Upsample(scale_factor=(1, 2))  # (2, 1, 128)
        self.inv_conv2 = nn.ConvTranspose2d(2, 4, kernel_size=kernel_size, padding=(0, 2))  # (4, 1, 128)
        self.up2 = nn.Upsample(scale_factor=(1, 2))  # (4, 1, 256)
        self.inv_conv3 = nn.ConvTranspose2d(4, 32, kernel_size=kernel_size, padding=(0, 2))  # (32, 1, 256)
        self.up3 = nn.Upsample(scale_factor=(1, 2))  # (32, 1, 512)
        self.inv_conv4 = nn.ConvTranspose2d(32, 64, kernel_size=kernel_size, padding=(0, 2))  # (64, 1, 512)
        self.up4 = nn.Upsample(scale_factor=(1, 2))  # (64, 1, 1024)
        self.inv_conv5 = nn.ConvTranspose2d(64, 1, kernel_size=kernel_size, padding=(0, 2))  # (1, 1, 1024)

    def encoder(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), (1, 2))  # (64, 1, 512)
        x = F.avg_pool2d(F.relu(self.conv2(x)), (1, 2))  # (32, 1, 256)
        x = F.avg_pool2d(F.relu(self.conv3(x)), (1, 2))  # (4, 1, 128)
        x = F.avg_pool2d(F.relu(self.conv4(x)), (1, 2))  # (2, 1, 64)

        return x

    def decoder(self, x):
        x = self.up1(F.relu(self.inv_conv1(x)))
        x = self.up2(F.relu(self.inv_conv2(x)))
        x = self.up3(F.relu(self.inv_conv3(x)))
        x = self.up4(F.relu(self.inv_conv4(x)))
        x = self.inv_conv5(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


######################################################
######################################################


class VAE(nn.Module):

    def __init__(self, h_dim=128, z_dim=32 ):
        super(VAE, self).__init__()
        # input size = (batch_size, 1, 1, 1024)
        # kernel
        kernel_size = (1, 5)
        # encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, padding=(0, 2))  # (64, 1, 1024)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel_size, padding=(0, 2))  # (32, 1, 512)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=kernel_size, padding=(0, 2))  # (4, 1, 256)
        self.conv4 = nn.Conv2d(4, 2, kernel_size=kernel_size, padding=(0, 2))  # (2, 1, 128)
        self.flatten = nn.Flatten(1, -1)  # (128)

        # bottleneck

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # decoder
        self.unflatten = nn.Unflatten(1, (2, 1, 64))
        self.inv_conv1 = nn.ConvTranspose2d(2, 2, kernel_size=kernel_size, padding=(0, 2))  # (2, 1, 64)
        self.up1 = nn.Upsample(scale_factor=(1, 2))  # (2, 1, 128)
        self.inv_conv2 = nn.ConvTranspose2d(2, 4, kernel_size=kernel_size, padding=(0, 2))  # (4, 1, 128)
        self.up2 = nn.Upsample(scale_factor=(1, 2))  # (4, 1, 256)
        self.inv_conv3 = nn.ConvTranspose2d(4, 32, kernel_size=kernel_size, padding=(0, 2))  # (32, 1, 256)
        self.up3 = nn.Upsample(scale_factor=(1, 2))  # (32, 1, 512)
        self.inv_conv4 = nn.ConvTranspose2d(32, 64, kernel_size=kernel_size, padding=(0, 2))  # (64, 1, 512)
        self.up4 = nn.Upsample(scale_factor=(1, 2))  # (64, 1, 1024)
        self.inv_conv5 = nn.ConvTranspose2d(64, 1, kernel_size=kernel_size, padding=(0, 2))  # (1, 1, 1024)

    def encoder(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), (1, 2))  # (64, 1, 512)
        x = F.avg_pool2d(F.relu(self.conv2(x)), (1, 2))  # (32, 1, 256)
        x = F.avg_pool2d(F.relu(self.conv3(x)), (1, 2))  # (4, 1, 128)
        x = F.avg_pool2d(F.relu(self.conv4(x)), (1, 2))  # (2, 1, 64)
        x = self.flatten(x)

        return x

    def decoder(self, x):
        x = self.unflatten(x)
        x = self.up1(F.relu(self.inv_conv1(x)))
        x = self.up2(F.relu(self.inv_conv2(x)))
        x = self.up3(F.relu(self.inv_conv3(x)))
        x = self.up4(F.relu(self.inv_conv4(x)))
        x = self.inv_conv5(x)

        return x

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

net = VAE()
summary(net, input_size=(1, 1, 1024))
print(net)