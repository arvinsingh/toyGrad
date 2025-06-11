import random

from .module import Module, ModuleList
from .special import SScalar



class Unit(Module):
    def __init__(self, n_units):
        super().__init__()
        self.weights = [SScalar(random.uniform(-1.0, 1.0)) for _ in range(n_units)]
        self.bias = SScalar(random.uniform(-1.0, 1.0))
    
    def __call__(self, x):
        act = sum(((weight_i * x_i) for weight_i, x_i in zip(self.weights, x)), self.bias)
        out = act.tanh()
        return out


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.units = [Unit(in_features) for _ in range(out_features)]

    def __call__(self, x):
        outs = [unit(x) for unit in self.units]
        return outs[0] if len(outs) == 1 else outs


class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super().__init__()
        self.layers = ModuleList()

        # i/p
        self.layers.append(Linear(input_size, hidden_size))

        # dynamically created hidden layers
        for _ in range(hidden_layers):
            self.layers.append(Linear(hidden_size, hidden_size))

        # o/p
        self.layers.append(Linear(hidden_size, output_size))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)  # no activation on o/p layer
        return x


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()

