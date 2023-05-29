import sys
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch.nn.functional import softmax
import numpy as np
from numpy.typing import ArrayLike

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../KENN-Pytorch/')

from kenn.parsers import relational_parser
from kenn.boost_functions.boost import GodelBoostResiduum, GodelBoostResiduumApprox


class MLP_within2(nn.Module):
    def __init__(self, input_shape: int):
        super(MLP_within2, self).__init__()
        self.h1 = Linear(input_shape, 1024)
        self.d1 = Dropout(0.5)
        self.last_layer = Linear(1024, 4)

    def preactivations(self, inputs: torch.Tensor):
        x = torch.relu(self.h1(inputs))
        x = self.d1(x)

        return self.last_layer(x)

    def forward(self, inputs: torch.Tensor, **kwargs):
        z = self.preactivations(inputs)

        return z, softmax(z, dim=1)

class Kenn2(nn.Module):

    def __init__(self, knowledge_file: str, input_shape: int, nnDistance, nnOcclusion, n_layers: int, boost_function=GodelBoostResiduumApprox):
        super(Kenn2, self).__init__()
        self.knowledge = knowledge_file
        self.kenn_layers = []
        self.nn_d = nnDistance
        self.nn_o = nnOcclusion

        for _ in range(n_layers):
            self.kenn_layers.append(relational_parser(self.knowledge, boost_function=boost_function))

    def forward(self, inputs: [torch.Tensor, ArrayLike, torch.Tensor, torch.Tensor], **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        d, _ = self.nn_d(features)
        o, _ = self.nn_o(features)

        ymin_diff = features[:, 2] - features[:, -2]
        intersection_mask = ((features[:, 0] <= features[:, 5]) & (features[:, 1] >= features[:, 4]) & (features[:, 2] <= features[:, -1]) & (features[:, 3] >= features[:, -2]))
        intersection = torch.Tensor(intersection_mask * 10) - 5
        z = torch.cat((d, o, torch.unsqueeze(ymin_diff, dim=1), torch.unsqueeze(intersection, dim=1)), 1)

        for ke in self.kenn_layers:
            z, _ = ke(z, relations, sx, sy)

        z = z[:, :-2]
        return z[:, :4], softmax(z[:, :4], dim=1), z[:, 4:], softmax(z[:, 4:], dim=1)

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

    def __init__(self, knowledge_file: str, input_shape: int, n_layers: int, boost_function=GodelBoostResiduum):
        super(Kenn_across, self).__init__(input_shape)
        self.knowledge = knowledge_file
        self.kenn_layers = []

        for _ in range(n_layers):
            self.kenn_layers.append(relational_parser(self.knowledge, boost_function=boost_function))

    def forward(self, inputs: [torch.Tensor, torch.Tensor, ArrayLike, ArrayLike, ArrayLike], **kwargs):
        features = inputs[0]
        within_preactivations = inputs[1]
        index_xy = inputs[2]
        index_yz = inputs[3]
        index_xz = inputs[4]

        #within_preactivations = np.where((within_preactivations[:, 1] > within_preactivations[:, 2]), within_preactivations[:, 1].detach().numpy(), -within_preactivations[:, 2].detach().numpy())
        within_preactivations = torch.tensor(within_preactivations.detach().numpy())
        across_preactivations = self.preactivations(features)
        preactivations = torch.cat([within_preactivations[:, :3], across_preactivations], 0)

        if index_xz and index_yz and index_xz:
            xy = preactivations[index_xy]
            yz = preactivations[index_yz]
            xz = preactivations[index_xz]
            unary = torch.cat([xy, yz, xz], dim=1)

            for ke in self.kenn_layers:
                unary, _ = ke(unary, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

            preactivations[index_xz] = unary[:, 6:]

        return preactivations[within_preactivations.shape[0]:], softmax(preactivations[within_preactivations.shape[0]:], dim=1)


    class Kenn_across_groundtruth(MLP_across):

        def __init__(self, knowledge_file: str, input_shape: int, n_layers: int,
                     boost_function=GodelBoostResiduum):
            super(Kenn_across_groundtruth, self).__init__(input_shape)
            self.knowledge = knowledge_file
            self.kenn_layers = []

            for _ in range(n_layers):
                self.kenn_layers.append(relational_parser(self.knowledge, boost_function=boost_function))

        def forward(self, inputs: [torch.Tensor, torch.Tensor, ArrayLike, ArrayLike, ArrayLike], **kwargs):
            features = inputs[0]
            within_predictions = inputs[1]
            index_xy = inputs[2]
            index_yz = inputs[3]
            index_xz = inputs[4]

            within_predictions = torch.eye(4)[within_predictions] * 500
            # within_preactivations = np.where((within_preactivations[:, 1] > within_preactivations[:, 2]), within_preactivations[:, 1].detach().numpy(), -within_preactivations[:, 2].detach().numpy())
            within_predictions = torch.tensor(within_predictions.detach().numpy())
            across_preactivations = self.preactivations(features)
            preactivations = torch.cat([within_predictions[:, :3], across_preactivations], 0)

            if index_xz and index_yz and index_xz:
                xy = preactivations[index_xy]
                yz = preactivations[index_yz]
                xz = preactivations[index_xz]
                unary = torch.cat([xy, yz, xz], dim=1)

                for ke in self.kenn_layers:
                    unary, _ = ke(unary, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

                preactivations[index_xz] = unary[:, 6:]

            return preactivations[within_predictions.shape[0]:], softmax(preactivations[within_predictions.shape[0]:],
                                                                         dim=1)

