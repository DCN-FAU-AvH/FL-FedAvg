from torch import nn
import torch.nn.functional as F


class MLP_1NN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_1NN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP_2NN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP_2NN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_M(nn.Module):
    def __init__(self, args, dim_channels, dim_fc_hiddens):
        super(CNN_M, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=args.num_channels, out_channels=dim_channels, kernel_size=5
        )
        self.conv2 = nn.Conv2d(
            in_channels=dim_channels, out_channels=dim_channels * 2, kernel_size=5
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=(dim_channels * 2) * 4 * 4, out_features=dim_fc_hiddens
        )
        self.fc2 = nn.Linear(in_features=dim_fc_hiddens, out_features=args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class ResidualBlock_M(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=5, padding=2)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class ResNet_M(nn.Module):
    def __init__(self, args, dim_channels, dim_fc_hiddens):
        super().__init__()
        self.conv1 = nn.Conv2d(args.num_channels, dim_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(dim_channels, dim_channels * 2, kernel_size=5)
        self.res_block_1 = ResidualBlock_M(dim_channels)
        self.res_block_2 = ResidualBlock_M(dim_channels * 2)
        self.fc1 = nn.Linear(16 * dim_channels * 2, dim_fc_hiddens)
        self.fc2 = nn.Linear(dim_fc_hiddens, args.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.res_block_1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.res_block_2(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
