from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch import flatten


class ZooNet(Module):
    '''
        Params:
            - num_channels: Number of channels in the input images. 3 for RGB, 1 for grayscale.
            - classes: Total number of unique classes in our dataset.

    '''

    def __init__(self, num_channels, classes):
        super(ZooNet, self).__init__()
        # CONV => RELU => POOL
        self.conv_1 = Conv2d(in_channels=num_channels,
                             out_channels=10, kernel_size=(3, 3))
        self.relu_1 = ReLU()
        self.maxpool_1 = MaxPool2d(kernel_size=(5, 5), stride=(2, 2))

        # CONV => RELU => POOL
        self.conv_2 = Conv2d(
            in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.relu_2 = ReLU()
        self.maxpool_2 = MaxPool2d(kernel_size=(5, 5), stride=(2, 2))

        # # CONV => RELU => POOL
        # self.conv_2 = Conv2d(
        #     in_channels=10, out_channels=20, kernel_size=(3, 3))
        # self.relu_2 = ReLU()
        # self.maxpool_2 = MaxPool2d(kernel_size=(5, 5), stride=(2, 2))

        # FULLY CONNECTED => RELU
        self.fc_1 = Linear(in_features=15680 , out_features=500)
        self.relu_3 = ReLU()

        # Softmax
        self.fc_2 = Linear(in_features=500, out_features=classes)
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc_1(x)
        x = self.relu_3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc_2(x)
        output = self.log_softmax(x)
        # return the output predictions
        return output
