import torch.nn as nn
import torch.nn.functional as F
import torch

from torchsummary import summary

################################
#                              #
#           MODEL 1D           #
#     Fixed Kernel Size        #
#                              #
################################


class AutoEncoder_1D(nn.Module):

    def __init__(self, kernel_size, padding):
        super(AutoEncoder_1D, self).__init__()
        # input size = (batch_size, 2, len_sample=11000)

        # encoder
        self.pad = nn.ZeroPad2d(padding=(4, 4, 0, 0))  # (2, 11008)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=kernel_size, padding=padding)  # (64, 11008)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=kernel_size, padding=padding)  # (32, 5504)
        self.conv3 = nn.Conv1d(32, 4, kernel_size=kernel_size, padding=padding)  # (4, 2752)
        self.conv4 = nn.Conv1d(4, 2, kernel_size=kernel_size, padding=padding)  # (2, 1376)

        # decoder
        self.inv_conv1 = nn.ConvTranspose1d(2, 2, kernel_size=kernel_size, padding=padding)  # (2, 688)
        self.up1 = nn.Upsample(scale_factor=2)  # (2, 1376)
        self.inv_conv2 = nn.ConvTranspose1d(2, 4, kernel_size=kernel_size, padding=padding)  # (4, 1376)
        self.up2 = nn.Upsample(scale_factor=2)  # (4, 2752)
        self.inv_conv3 = nn.ConvTranspose1d(4, 32, kernel_size=kernel_size, padding=padding)  # (32, 2752)
        self.up3 = nn.Upsample(scale_factor=2)  # (32, 5504)
        self.inv_conv4 = nn.ConvTranspose1d(32, 64, kernel_size=kernel_size, padding=padding)  # (64, 5504)
        self.up4 = nn.Upsample(scale_factor=2)  # (64, 11008)
        self.inv_conv5 = nn.ConvTranspose1d(64, 2, kernel_size=kernel_size, padding=padding)  # (2, 11008)

    def encoder(self, x):
        # print(x.shape)
        x = self.pad(x)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=(1, 2))  # (64, 5504)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv2(x)), (1, 2))  # (32, 2752)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv3(x)), (1, 2))  # (4, 1376)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv4(x)), (1, 2))  # (2, 688)
        # print(x.shape)

        return x

    def decoder(self, x):
        # print(x.shape)
        x = self.up1(F.relu(self.inv_conv1(x)))
        # print(x.shape)
        x = self.up2(F.relu(self.inv_conv2(x)))
        # print(x.shape)
        x = self.up3(F.relu(self.inv_conv3(x)))
        # print(x.shape)
        x = self.up4(F.relu(self.inv_conv4(x)))
        # print(x.shape)
        x = self.inv_conv5(x)
        # print(x.shape)
        x = x[:, :, 4: 11004]
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


################################
#                              #
#           MODEL 1D           #
#     Various Kernel Size      #
#                              #
################################


class AutoEncoder_1D_VarKernel(nn.Module):

    def __init__(self, kernel_size, padding):
        super(AutoEncoder_1D_VarKernel, self).__init__()
        # input size = (batch_size, 2, len_sample=11000)

        # encoder
        self.pad = nn.ZeroPad2d(padding=(4, 4, 0, 0))  # (2, 11008)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=kernel_size[0], padding=padding[0])  # (64, 11008)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=kernel_size[1], padding=padding[1])  # (32, 5504)
        self.conv3 = nn.Conv1d(32, 4, kernel_size=kernel_size[2], padding=padding[2])  # (4, 2752)
        self.conv4 = nn.Conv1d(4, 2, kernel_size=kernel_size[3], padding=padding[3])  # (2, 1376)

        # decoder
        self.inv_conv1 = nn.ConvTranspose1d(2, 2, kernel_size=kernel_size[3], padding=padding[3])  # (2, 688)
        self.up1 = nn.Upsample(scale_factor=2)  # (2, 1376)
        self.inv_conv2 = nn.ConvTranspose1d(2, 4, kernel_size=kernel_size[2], padding=padding[2])  # (4, 1376)
        self.up2 = nn.Upsample(scale_factor=2)  # (4, 2752)
        self.inv_conv3 = nn.ConvTranspose1d(4, 32, kernel_size=kernel_size[1], padding=padding[1])  # (32, 2752)
        self.up3 = nn.Upsample(scale_factor=2)  # (32, 5504)
        self.inv_conv4 = nn.ConvTranspose1d(32, 64, kernel_size=kernel_size[0], padding=padding[0])  # (64, 5504)
        self.up4 = nn.Upsample(scale_factor=2)  # (64, 11008)
        self.inv_conv5 = nn.ConvTranspose1d(64, 2, kernel_size=kernel_size[0], padding=padding[0])  # (2, 11008)

    def encoder(self, x):
        # print(x.shape)
        x = self.pad(x)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=(1, 2))  # (64, 5504)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv2(x)), (1, 2))  # (32, 2752)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv3(x)), (1, 2))  # (4, 1376)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv4(x)), (1, 2))  # (2, 688)
        # print(x.shape)

        return x

    def decoder(self, x):
        # print(x.shape)
        x = self.up1(F.relu(self.inv_conv1(x)))
        # print(x.shape)
        x = self.up2(F.relu(self.inv_conv2(x)))
        # print(x.shape)
        x = self.up3(F.relu(self.inv_conv3(x)))
        # print(x.shape)
        x = self.up4(F.relu(self.inv_conv4(x)))
        # print(x.shape)
        x = self.inv_conv5(x)
        # print(x.shape)
        x = x[:, :, 4: 11004]
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


################################
#                              #
#           MODEL 2D           #
#                              #
################################


class AutoEncoder_2D(nn.Module):

    def __init__(self, kernel_size=(3, 5)):
        super(AutoEncoder_2D, self).__init__()
        # input size = (batch_size, 1, 2, len_sample=11000)

        # encoder
        self.pad = nn.ZeroPad2d(padding=(4, 4, 0, 0))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, padding=(1, 2))  # (64, 2, len_sample)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel_size, padding=(1, 2))  # (32, 2, 5500)
        self.conv3 = nn.Conv2d(32, 4, kernel_size=kernel_size, padding=(1, 2))  # (4, 2, 2750)
        self.conv4 = nn.Conv2d(4, 2, kernel_size=kernel_size, padding=(1, 2))  # (2, 2, 1375)

        # decoder
        self.inv_conv1 = nn.ConvTranspose2d(2, 2, kernel_size=kernel_size, padding=(1, 2))  # (2, 2, 687)
        self.up1 = nn.Upsample(scale_factor=(1, 2))  # (2, 1, 1375)
        self.inv_conv2 = nn.ConvTranspose2d(2, 4, kernel_size=kernel_size, padding=(1, 2))  # (4, 2, 1375)
        self.up2 = nn.Upsample(scale_factor=(1, 2))  # (4, 1, 2750)
        self.inv_conv3 = nn.ConvTranspose2d(4, 32, kernel_size=kernel_size, padding=(1, 2))  # (32, 2, 2750)
        self.up3 = nn.Upsample(scale_factor=(1, 2))  # (32, 1, 5500)
        self.inv_conv4 = nn.ConvTranspose2d(32, 64, kernel_size=kernel_size, padding=(1, 2))  # (64, 2, 5500)
        self.up4 = nn.Upsample(scale_factor=(1, 2))  # (64, 1, 11000)
        self.inv_conv5 = nn.ConvTranspose2d(64, 2, kernel_size=kernel_size, padding=(1, 2))  # (1, 2, len_sample)

    def encoder(self, x):
        # print(x.shape)
        x = self.pad(x)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv1(x)), kernel_size=(1, 2))  # (64, 2, 5500)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv2(x)), (1, 2))  # (32, 2, 2750)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv3(x)), (1, 2))  # (4, 2, 1375)
        # print(x.shape)
        x = F.avg_pool2d(F.relu(self.conv4(x)), (1, 2))  # (2, 2, 687)
        # print(x.shape)

        return x

    def decoder(self, x):
        # print(x.shape)
        x = self.up1(F.relu(self.inv_conv1(x)))
        # print(x.shape)
        x = self.up2(F.relu(self.inv_conv2(x)))
        # print(x.shape)
        x = self.up3(F.relu(self.inv_conv3(x)))
        # print(x.shape)
        x = self.up4(F.relu(self.inv_conv4(x)))
        # print(x.shape)
        x = self.inv_conv5(x)
        # print(x.shape)
        x = x[:, :, :, 4: 11004]
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

################################
#                              #
#             MAIN             #
#                              #
################################


if __name__ == "__main__":
    net = AutoEncoder_1D_VarKernel(kernel_size=[13, 11, 7, 5], padding=[6, 5, 3, 2])
    summary(net, input_size=(2, 11000))
    print(net)
