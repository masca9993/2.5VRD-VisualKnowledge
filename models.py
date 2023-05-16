import sys
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch.nn.functional import softmax
from numpy.typing import ArrayLike

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../KENN-Pytorch/')

from kenn.parsers import relational_parser
from kenn.boost_functions.boost import GodelBoostResiduum, GodelBoostResiduumApprox


class MLP_within(nn.Module):
    def __init__(self, input_shape: int):
        super(MLP_within, self).__init__()
        self.h1 = Linear(input_shape, 1024)
        self.d1 = Dropout(0.5)
        self.last_layer = Linear(1024, 8)

    def preactivations(self, inputs: torch.Tensor):
        x = torch.relu(self.h1(inputs))
        x = self.d1(x)

        return self.last_layer(x)

    def forward(self, inputs: torch.Tensor, **kwargs):
        z = self.preactivations(inputs)

        return z[:, :4], softmax(z[:, :4], dim=1), z[:, 4:], softmax(z[:, 4:], dim=1)


class MLP_across(nn.Module):
    def __init__(self, input_shape: int):
        super(MLP_across, self).__init__()
        self.h1 = Linear(input_shape, 1024)
        self.d1 = Dropout(0.5)
        self.last_layer = Linear(1024, 3)

    def preactivations(self, inputs: torch.Tensor):
        x = torch.relu(self.h1(inputs))
        x = self.d1(x)

        return self.last_layer(x)

    def forward(self, inputs: torch.Tensor, **kwargs):
        z = self.preactivations(inputs)

        return z, softmax(z, dim=1)


class Kenn(MLP_within):

    def __init__(self, knowledge_file: str, input_shape: int, n_layers: int, boost_function=GodelBoostResiduumApprox):
        super(Kenn, self).__init__(input_shape)
        self.knowledge = knowledge_file
        self.kenn_layers = []

        for _ in range(n_layers):
            self.kenn_layers.append(relational_parser(self.knowledge, boost_function=boost_function))

    def forward(self, inputs: [torch.Tensor, ArrayLike, torch.Tensor, torch.Tensor], **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        z = self.preactivations(features)

        ymin_diff = (features[:, 2] - features[:, -2]) * 10
        intersection_mask = ((features[:, 0] <= features[:, 5]) & (features[:, 1] >= features[:, 4]) & (features[:, 2] <= features[:, -1]) & (features[:, 3] >= features[:, -2]))
        intersection = torch.Tensor(intersection_mask * 10) - 5
        z = torch.cat((z, torch.unsqueeze(ymin_diff, dim=1), torch.unsqueeze(intersection, dim=1)), dim=1)

        for ke in self.kenn_layers:
            z, _ = ke(z, relations, sx, sy)

        z = z[:, :-2]
        return z[:, :4], softmax(z[:, :4], dim=1), z[:, 4:], softmax(z[:, 4:], dim=1)


class Kenn_across(MLP_across):

    def __init__(self, knowledge_file: str, input_shape: int, n_layers: int, boost_function=GodelBoostResiduumApprox):
        super(Kenn_across, self).__init__(input_shape)
        self.knowledge = knowledge_file
        self.kenn_layers = []

        for _ in range(n_layers):
            self.kenn_layers.append(relational_parser(self.knowledge, boost_function=boost_function))

    def forward(self, inputs: [torch.Tensor, ArrayLike, torch.Tensor, torch.Tensor], **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        z = self.preactivations(features)

        for ke in self.kenn_layers:
            z, _ = ke(z, relations, sx, sy)

        return z, softmax(z, dim=1)
