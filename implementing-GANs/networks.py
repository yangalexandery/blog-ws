import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, num_layers, latent_dim, channels):
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.channels = channels
        assert self.num_layers == len(self.channels)
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append([nn.ConvTranspose2d(self.latent_dim, self.channels[i], 4), nn.ReLU()])
            else:
                # self.features.append(self.upscale(self.channels[i-1])
                # self.features.append(F.upsample())
                self.layers.append([nn.Conv2d(self.channels[i-1], self.channels[i], 3, padding=1), nn.ReLU()])
        self.last_layer = [nn.Conv2d(self.channels[-1], self.channels[-1], 3, padding=1), nn.ReLU()]
        self.RGB_layer_1 = nn.Conv2d(self.channels[-1], 3, 1)
        self.RGB_layer_2 = nn.Conv2d(self.channels[-2], 3, 1)

    def forward(self, x):
        # latent vector x
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                x0 = x
            for module in self.layers[i]:
                x = module(x)
            if i != self.num_layers - 1:
                x = F.upsample(x, scale_factor=2, mode='nearest')
            else:
                for module in self.last_layer:
                    x = module(x)
        x = self.RGB_layer_1(x)
        if self.num_layers > 1:
            x0 = self.RGB_layer_2(x0)
            return self.smooth(x0, x)
        return x

    def smooth(self, x0, x1):
        return torch.add(torch.mul(x0, 1 - self.alpha), torch.mul(x1, self.alpha))


class Discriminator(nn.Module):

    def __init__(self, num_layers, channels):
        self.num_layers = num_layers
        self.channels = channels
        assert self.num_layers == len(self.channels)
        self.layers = []
        self.RGB_layer_1 = [nn.Conv2d(3, self.channels[-1], 1), nn.ReLU()]
        self.RGB_layer_2 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(3, self.channels[-2], 1), nn.ReLU()]
        for i in reversed(range(self.num_layers)):
            if i > 0:
                self.layers.append([nn.Conv2d(self.channels[i], self.channels[i-1], 3, padding=1), nn.ReLU(), nn.AvgPool2d(2, stride=2)])
                # this may need checking
            else:
                # need to do minibatch-stddev
                self.layers.append([nn.Conv2d(self.channels[0], self.channels[0], 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(self.channels[0], self.channels[0], 4), nn.ReLU()])

    def forward(self, x):
        x1 = x
        for module in self.RGB_layer_1:
            x1 = module(x1)
        x0 = x
        for module in self.RGB_layer_2:
            x0 = module(x0)
        for i in reversed(range(self.num_layers)):
            for module in layer:
                x1 = module(x1)
            x1 = self.smooth(x0, x1)
        return x1

    def smooth(self, x0, x1):
        return torch.add(torch.mul(x0, 1 - self.alpha), torch.mul(x1, self.alpha))



def FC(dim_list, input_dim = 32 * 32 * 3, output_dim = 10):
    """Constructs a fully connected network with given hidden dimensions"""
    modules = []

    for h_dim in dim_list:
        modules.append(nn.Linear(input_dim, h_dim))
        modules.append(nn.ELU())
        input_dim = h_dim

    modules.append(nn.Linear(input_dim, output_dim))

    return nn.Sequential(*modules)

class ConvNet(nn.Module):

    def __init__(self, output_dim=10):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 192, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 384, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
			# nn.Linear(128 * 7 * 7, 512),
			# nn.ELU(),
			# nn.Linear(512, output_dim)
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(384 * 8 * 8, 1024),
            nn.ELU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 384 * 8 * 8)
        x = self.classifier(x)
        return x

# def ConvNet(output_dim = 10):
# 	modules = [
# 		nn.Conv2d(3, 64, kernel_size=7),
# 		nn.ELU(),
# 		nn.MaxPool2d(kernel_size=2, stride=2),
# 		nn.Conv2d(64, 128, kernel_size=5),
# 		nn.ELU(),
# 		nn.Conv2d(128, 128, kernel_size=3, padding=1),
# 		nn.ELU(),
# 		nn.Linear(128 * 7 * 7, 512),
# 		nn.ELU(),
# 		nn.Linear(512, output_dim)
# 	]
# 	return nn.Sequential(*modules)
