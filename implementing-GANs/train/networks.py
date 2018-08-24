import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SN(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SN, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    
class Generator(nn.Module):

    def __init__(self, num_layers, latent_dim, channels):
        super(Generator, self).__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.channels = channels
        assert self.num_layers == len(self.channels)
        self.layers = []
        self.params = nn.ParameterList()

        for i in range(self.num_layers):
            if i == 0:
                self.layers.append([
                    SN(nn.ConvTranspose2d(self.latent_dim, self.channels[i], 4)), 
                    nn.LeakyReLU()
                ])
            else:
                # self.features.append(self.upscale(self.channels[i-1])
                # self.layers.append([F.upsample(scale_factor=2)])
                self.layers.append([
                    SN(nn.Conv2d(self.channels[i-1], self.channels[i], 3, padding=1)),
                    nn.LeakyReLU()
                ])

        self.last_layer = [
            SN(nn.Conv2d(self.channels[-1], self.channels[-1], 3, padding=1)),
            nn.LeakyReLU()
        ]

        self.RGB_layer_1 = SN(nn.Conv2d(self.channels[-1], 3, 1))
        self.RGB_layer_2 = [
            SN(nn.Conv2d(self.channels[-2], self.channels[-2], 3, padding=1)),
            nn.LeakyReLU(),
            nn.Conv2d(self.channels[-2], 3, 1)
        ] if self.num_layers > 1 else None

        self.alpha = 0
        self.find_params()


    def find_params(self):
        for layer in self.layers:
            for module in layer:
                for param in module.parameters():
                    self.params.append(param)
        for module in self.last_layer:
            for param in module.parameters():
                self.params.append(param)
        for param in self.RGB_layer_1.parameters():
            self.params.append(param)
        if self.RGB_layer_2:
            for module in self.RGB_layer_2:
                for param in module.parameters():
                    self.params.append(param)


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
            for module in self.RGB_layer_2:
                x0 = module(x0)
            return self.smooth(x0, x)
        return x


    def smooth(self, x0, x1):
        return torch.add(torch.mul(x0, 1 - self.alpha), torch.mul(x1, self.alpha))


    def load_prev_model(self, old_g):
        for i in range(len(old_g.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].load_state_dict(old_g.layers[i][j].state_dict())
        self.RGB_layer_2[0].load_state_dict(old_g.last_layer[0].state_dict())
        self.RGB_layer_2[2].load_state_dict(old_g.RGB_layer_1.state_dict())

        
class Discriminator(nn.Module):

    def __init__(self, num_layers, channels):
        super(Discriminator, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        assert self.num_layers == len(self.channels)
        self.layers = []
        self.params = nn.ParameterList()

        self.RGB_layer_1 = [
            SN(nn.Conv2d(3, self.channels[-1], 1)),
            nn.LeakyReLU()
        ]
        self.RGB_layer_2 = [
            nn.AvgPool2d(2, stride=2), 
            SN(nn.Conv2d(3, self.channels[-2], 1)),
            nn.LeakyReLU()
        ] if self.num_layers > 1 else None

        for i in reversed(range(self.num_layers)):
            if i > 0:
                self.layers.append([
                    SN(nn.Conv2d(self.channels[i], self.channels[i-1], 3, padding=1)),
                    nn.LeakyReLU(),
                    nn.AvgPool2d(2, stride=2)
                ])
                # this may need checking
            else:
                # need to do minibatch-stddev
                self.layers.append([
                    SN(nn.Conv2d(self.channels[0], self.channels[0], 3, padding=1)),
                    nn.LeakyReLU(),
                    SN(nn.Conv2d(self.channels[0], 1, 4))
                ])
        self.alpha = 0

        self.get_params()


    def get_params(self):
        for layer in self.layers:
            for module in layer:
                for param in module.parameters():
                    self.params.append(param)
        for module in self.RGB_layer_1:
            for param in module.parameters():
                self.params.append(param)
        if self.RGB_layer_2:
            for module in self.RGB_layer_2:
                for param in module.parameters():
                    self.params.append(param)


    def forward(self, x):
        x1 = x
        for module in self.RGB_layer_1:
            x1 = module(x1)
        x0 = x
        for i in range(self.num_layers):
            for module in self.layers[i]:
                x1 = module(x1)
            if i == 0 and self.RGB_layer_2:
                for module in self.RGB_layer_2:
                    x0 = module(x0)
                x1 = self.smooth(x0, x1)
        return x1


    def smooth(self, x0, x1):
        return torch.add(torch.mul(x0, 1 - self.alpha), torch.mul(x1, self.alpha))


    def load_prev_model(self, old_g):
        for i in range(len(old_g.layers)):
            for j in range(len(self.layers[-i-1])):
                self.layers[-i-1][j].load_state_dict(old_g.layers[-i-1][j].state_dict())
        self.RGB_layer_2[1].load_state_dict(old_g.RGB_layer_1[0].state_dict())


class Downsampler(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Downsampler, self).__init__()
        self.size = input_dim // output_dim
        self.layer = nn.AvgPool2d(self.size, stride=self.size)

    def forward(self, x):
        return self.layer(x)

    def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

