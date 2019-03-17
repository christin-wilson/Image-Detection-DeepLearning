import torch
import torch.nn as nn
import numpy as np

class SEC_NET(nn.Module):
    _net_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, input_channels=3, class_size=10, weights=True, batch_norm=True):
        super(SEC_NET, self).__init__()
        '''
        self.architecture = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        '''
        self.batch_norm = batch_norm

        net_layers = []

        for config in self._net_config:
            if config != 'M':
                conv = nn.Conv2d(input_channels, config, kernel_size=3, padding=1)
                if self.batch_norm:
                    net_layers += [conv, nn.BatchNorm2d(config), nn.ReLU(True)]
                else:
                    net_layers += [conv, nn.ReLU(True)]
                # set the input of the next convolution to the output of the previous convolution
                input_channels = config
            else:
                net_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        self.architecture = nn.Sequential(*net_layers)
        #self.average_pool = nn.AdaptiveAvgPool2d((7,7))
        self.fully_connected = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, class_size),
        )

        if weights:
            self._configure_weights()

    #initialize weights using He's technique
    def _configure_weights(self):
        for index, module in enumerate(self.modules()):
            # print(index, "-->", module)  #diplay all modules
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.architecture(x)
        #print(x.size(0))
        #print(x.size())
        #x = self.average_pool(x)
        x = x.view(x.size(0), -1)  #reshape, let row be the size of x and column inferred from dimension
        x = self.fully_connected(x)
        return x

