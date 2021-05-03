from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim, Tensor


class Net(nn.Module):
    @staticmethod
    def create_target_weights(input_weight_matrix: Tensor) -> Tensor:
        """ Outputting a tensor with the target weights. """

        target_weight_matrix = np.arange(len(input_weight_matrix)).reshape(len(input_weight_matrix), 1).astype("f")

        for i in range(len(input_weight_matrix)):
            target_weight_matrix[i] = input_weight_matrix[i][0]

        return torch.from_numpy(target_weight_matrix)

    @staticmethod
    def are_weights_diverged(network_weights):
        """ Testing if the weights are eiter converging to infinity or -infinity. """

        for layer_id, layer in enumerate(network_weights):
            for cell_id, cell in enumerate(layer):
                for weight_id, weight in enumerate(cell):
                    if np.isnan(weight):
                        return True
                    if np.isinf(weight):
                        return True
        return False

    @staticmethod
    def apply_weights(network: Net, new_weights: Tensor) -> Net:
        """ Changing the weights of a network to new given values. """

        i = 0

        for layer_id, layer_name in enumerate(network.state_dict()):
            for line_id, line_values in enumerate(network.state_dict()[layer_name]):
                for weight_id, weight_value in enumerate(network.state_dict()[layer_name][line_id]):
                    network.state_dict()[layer_name][line_id][weight_id] = new_weights[i]
                    i += 1

        return network

    def __init__(self, i_size: int, h_size: int, o_size: int, name=None) -> None:
        super().__init__()

        self.name = name
        self.input_size = i_size
        self.hidden_size = h_size
        self.out_size = o_size

        self.no_weights = h_size * (i_size + h_size * (h_size - 1) + o_size)

        """ Data saved in self.s_train_weights_history & self.s_application_weights_history is used for experiments. """
        self.s_train_weights_history = []
        self.s_application_weights_history = []
        self.loss_history = []
        self.trained = False

        self.is_fixpoint = ""

        self.fc1 = nn.Linear(i_size, h_size, False)
        self.fc2 = nn.Linear(h_size, h_size, False)
        self.fc3 = nn.Linear(h_size, o_size, False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def normalize(self, value):
        """ Normalizing the values >= 1 and adding pow(10, -8) to the values equal to 0 """

        if value >= 1:
            return value/len(self.state_dict())
        elif value == 0:
            return pow(10, -8)
        else:
            return value

    def input_weight_matrix(self) -> Tensor:
        """ Calculating the input tensor formed from the weights of the net """

        # The "4" represents the weightwise coordinates used for the matrix: <value><layer_id><cell_id><positional_id>
        weight_matrix = np.arange(self.no_weights * 4).reshape(self.no_weights, 4).astype("f")

        i = 0

        for layer_id, layer_name in enumerate(self.state_dict()):
            for line_id, line_values in enumerate(self.state_dict()[layer_name]):
                for weight_id, weight_value in enumerate(self.state_dict()[layer_name][line_id]):
                    weight_matrix[i] = weight_value.item(), self.normalize(layer_id), self.normalize(weight_id), self.normalize(line_id)
                    i += 1

        return torch.from_numpy(weight_matrix)

    def self_train(self, training_steps: int, log_step_size: int, learning_rate: float, input_data: Tensor, target_data: Tensor) -> (np.ndarray, Tensor, list):
        """ Training a network to predict its own weights in order to self-replicate. """

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.trained = True

        for training_step in range(training_steps):
            output = self(input_data)
            loss = F.mse_loss(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Saving the history of the weights after a certain amount of steps (aka log_step_size) for research.
            # If it is a soup/mixed env. save weights only at the end of all training steps (aka a soup/mixed epoch)
            if "soup" not in self.name and "mixed" not in self.name:
                # If self-training steps are lower than 10, then append weight history after each ST step.
                if training_steps < 10:
                    self.s_train_weights_history.append(output.T.detach().numpy())
                    self.loss_history.append(round(loss.detach().numpy().item(), 5))
                else:
                    if training_step % log_step_size == 0:
                        self.s_train_weights_history.append(output.T.detach().numpy())
                        self.loss_history.append(round(loss.detach().numpy().item(), 5))

        # Saving weights only at the end of a soup/mixed exp. epoch.
        if "soup" in self.name or "mixed" in self.name:
            self.s_train_weights_history.append(output.T.detach().numpy())
            self.loss_history.append(round(loss.detach().numpy().item(), 5))

        return output.detach().numpy(), loss, self.loss_history

    def self_application(self, weights_matrix: Tensor, SA_steps: int, log_step_size: int) -> Net:
        """ Inputting the weights of a network to itself for a number of steps, without backpropagation. """

        data = copy.deepcopy(weights_matrix)
        new_net = copy.deepcopy(self)
        # output = new_net(data)

        for i in range(SA_steps):
            output = new_net(data)

            # Saving the weights history after a certain amount of steps (aka log_step_size) for research purposes.
            # If self-application steps are lower than 10, then append weight history after each SA step.
            if SA_steps < 10:
                self.s_application_weights_history.append(output.T.detach().numpy())
            else:
                if i % log_step_size == 0:
                    self.s_application_weights_history.append(output.T.detach().numpy())

            """ See after how many steps of SA is the output not changing anymore: """
            # print(f"Self-app. step {i+1}: {Experiment.changing_rate(output2, output)}")

            for j in range(len(data)):
                """ Constructing the weight matrix to have it as the next input. """
                data[j][0] = output[j]

            new_net = self.apply_weights(new_net, output)

        return new_net

    def attack(self, other_net: Net) -> Net:
        other_net_weights = other_net.input_weight_matrix()
        SA_steps = 1
        log_step_size = 1

        return self.self_application(other_net_weights, SA_steps, log_step_size)
