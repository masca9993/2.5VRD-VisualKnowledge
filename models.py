import sys
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch.nn.functional import softmax
import numpy as np
from numpy.typing import ArrayLike
import torch.nn.functional as F

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


    def forward(self, inputs: [torch.Tensor, torch.Tensor], **kwargs):
        features = inputs[0]
        intersections = inputs[1]

        d, _ = self.nn_d(features)
        o = torch.zeros([features.shape[0], 4])

        ymin_diff = features[:, 2] - features[:, -2]
        z = torch.cat((d, o, torch.unsqueeze(ymin_diff, dim=1), torch.unsqueeze(intersections, dim=1)), 1)

        for ke in self.kenn_layers:
            z, _ = ke(z, torch.Tensor([]), torch.Tensor([]), torch.Tensor([]))

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


class ImageEmbeddingCNN(nn.Module):
    def __init__(self):
        super(ImageEmbeddingCNN, self).__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=9, stride=4)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.fc = nn.Linear(1536, 16)

    def forward(self, x1, x2):  # Modify forward method to accept two input images
        x1 = self.avgpool1(x1)
        x1 = F.relu(self.conv1(x1))

        x1 = F.relu(self.conv2(x1))
        x1 = self.avgpool2(x1)

        x1 = x1.view(x1.size(0), -1)
        x1 = F.relu(self.fc(x1))

        x2 = self.avgpool1(x2)
        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))
        x2 = self.avgpool2(x2)

        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc(x2))

        return torch.cat((x1, x2), dim=1)


class MLP_SiftDepth(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_SiftDepth, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Combined model
class CombinedModel(nn.Module):
    def __init__(self, cnn, mlp):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.mlp = mlp

    def forward(self, images, features):
        embeddings = self.cnn(images[0], images[1])

        inputs = torch.cat((embeddings, features), dim=1)
        outputs = self.mlp(inputs)
        return outputs[:, :4], softmax(outputs[:, :4], dim=1), outputs[:, 4:], softmax(outputs[:, 4:], dim=1)


