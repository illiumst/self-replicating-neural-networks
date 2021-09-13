# from __future__ import annotations
import copy
import inspect
import random
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim, Tensor


def prng():
    return random.random()


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

    def apply_weights(self, new_weights: Tensor):
        """ Changing the weights of a network to new given values. """
        i = 0
        for layer_id, layer_name in enumerate(self.state_dict()):
            for line_id, line_values in enumerate(self.state_dict()[layer_name]):
                for weight_id, weight_value in enumerate(self.state_dict()[layer_name][line_id]):
                    self.state_dict()[layer_name][line_id][weight_id] = new_weights[i]
                    i += 1

        return self

    def __init__(self, i_size: int, h_size: int, o_size: int, name=None, start_time=1) -> None:
        super().__init__()
        self.start_time = start_time

        self.name = name
        self.children = []

        self.input_size = i_size
        self.hidden_size = h_size
        self.out_size = o_size

        self.no_weights = h_size * (i_size + h_size * (h_size - 1) + o_size)

        """ Data saved in self.s_train_weights_history & self.s_application_weights_history is used for experiments. """
        self.s_train_weights_history = []
        self.s_application_weights_history = []
        self.loss_history = []
        self.trained = False
        self.number_trained = 0

        self.is_fixpoint = ""

        self.fc1 = nn.Linear(i_size, h_size, False)
        self.fc2 = nn.Linear(h_size, h_size, False)
        self.fc3 = nn.Linear(h_size, o_size, False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def normalize(self, value, norm):
        """ Normalizing the values >= 1 and adding pow(10, -8) to the values equal to 0 """

        if norm > 1:
            return float(value) / float(norm)
        else:
            return float(value)

    def input_weight_matrix(self) -> Tensor:
        """ Calculating the input tensor formed from the weights of the net """

        # The "4" represents the weightwise coordinates used for the matrix: <value><layer_id><cell_id><positional_id>
        weight_matrix = np.arange(self.no_weights * 4).reshape(self.no_weights, 4).astype("f")

        i = 0
        max_layer_id = len(self.state_dict()) - 1
        for layer_id, layer_name in enumerate(self.state_dict()):
            max_cell_id = len(self.state_dict()[layer_name]) - 1
            for line_id, line_values in enumerate(self.state_dict()[layer_name]):
                max_weight_id = len(line_values) - 1
                for weight_id, weight_value in enumerate(self.state_dict()[layer_name][line_id]):
                    weight_matrix[i] = weight_value.item(), self.normalize(layer_id, max_layer_id), self.normalize(line_id, max_cell_id), self.normalize(weight_id, max_weight_id)
                    i += 1

        return torch.from_numpy(weight_matrix)

    def self_train(self, training_steps: int, log_step_size: int, learning_rate: float) -> (np.ndarray, Tensor, list):
        """ Training a network to predict its own weights in order to self-replicate. """

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

        for training_step in range(training_steps):
            self.number_trained += 1
            optimizer.zero_grad()
            input_data = self.input_weight_matrix()
            target_data = self.create_target_weights(input_data)
            output = self(input_data)
            loss = F.mse_loss(output, target_data)
            loss.backward()
            optimizer.step()

            # Saving the history of the weights after a certain amount of steps (aka log_step_size) for research.
            # If it is a soup/mixed env. save weights only at the end of all training steps (aka a soup/mixed epoch)
            if "soup" not in self.name and "mixed" not in self.name:
                weights = self.create_target_weights(self.input_weight_matrix())
                # If self-training steps are lower than 10, then append weight history after each ST step.
                if self.number_trained < 10:
                    self.s_train_weights_history.append(weights.T.detach().numpy())
                    self.loss_history.append(loss.detach().numpy().item())
                else:
                    if self.number_trained % log_step_size == 0:
                        self.s_train_weights_history.append(weights.T.detach().numpy())
                        self.loss_history.append(loss.detach().numpy().item())

        weights = self.create_target_weights(self.input_weight_matrix())
        # Saving weights only at the end of a soup/mixed exp. epoch.
        if "soup" in self.name or "mixed" in self.name:
            self.s_train_weights_history.append(weights.T.detach().numpy())
            self.loss_history.append(loss.detach().numpy().item())

        self.trained = True
        return weights.detach().numpy(), loss, self.loss_history

    def self_application(self, SA_steps: int, log_step_size: Union[int, None] = None):
        """ Inputting the weights of a network to itself for a number of steps, without backpropagation. """

        for i in range(SA_steps):
            output = self(self.input_weight_matrix())

            # Saving the weights history after a certain amount of steps (aka log_step_size) for research purposes.
            # If self-application steps are lower than 10, then append weight history after each SA step.
            if SA_steps < 10:
                weights = self.create_target_weights(self.input_weight_matrix())
                self.s_application_weights_history.append(weights.T.detach().numpy())
            else:
                weights = self.create_target_weights(self.input_weight_matrix())
                if i % log_step_size == 0:
                    self.s_application_weights_history.append(weights.T.detach().numpy())

            """ See after how many steps of SA is the output not changing anymore: """
            # print(f"Self-app. step {i+1}: {Experiment.changing_rate(output2, output)}")

            _ = self.apply_weights(output)

        return self

    def attack(self, other_net):
        other_net_weights = other_net.input_weight_matrix()
        my_evaluation = self(other_net_weights)
        return other_net.apply_weights(my_evaluation)

    def melt(self, other_net):
        try:
            melted_name = self.name + other_net.name
        except AttributeError:
            melted_name = None
        melted_weights = self.create_target_weights(other_net.input_weight_matrix())
        self_weights = self.create_target_weights(self.input_weight_matrix())
        weight_indxs = list(range(len(self_weights)))
        random.shuffle(weight_indxs)
        for weight_idx in weight_indxs[:len(melted_weights) // 2]:
            melted_weights[weight_idx] = self_weights[weight_idx]
        melted_net = Net(i_size=self.input_size, h_size=self.hidden_size, o_size=self.out_size, name=melted_name)
        melted_net.apply_weights(melted_weights)
        return melted_net

    def apply_noise(self, noise_size: float):
        """ Changing the weights of a network to values + noise """
        for layer_id, layer_name in enumerate(self.state_dict()):
            for line_id, line_values in enumerate(self.state_dict()[layer_name]):
                for weight_id, weight_value in enumerate(self.state_dict()[layer_name][line_id]):
                    # network.state_dict()[layer_name][line_id][weight_id] = weight_value + noise
                    if prng() < 0.5:
                        self.state_dict()[layer_name][line_id][weight_id] = weight_value + noise_size * prng()
                    else:
                        self.state_dict()[layer_name][line_id][weight_id] = weight_value - noise_size * prng()

        return self


class SecondaryNet(Net):

    def self_train(self, training_steps: int, log_step_size: int, learning_rate: float) -> (np.ndarray, Tensor, list):
        """ Training a network to predict its own weights in order to self-replicate. """

        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        df = pd.DataFrame(columns=['step', 'loss', 'first_to_target_loss', 'second_to_target_loss', 'second_to_first_loss'])
        is_diverged = False
        for training_step in range(training_steps):
            self.number_trained += 1
            optimizer.zero_grad()
            input_data = self.input_weight_matrix()
            target_data = self.create_target_weights(input_data)

            intermediate_output = self(input_data)
            second_input = copy.deepcopy(input_data)
            second_input[:, 0] = intermediate_output.squeeze()

            output = self(second_input)
            second_to_target_loss = F.mse_loss(output, target_data)
            first_to_target_loss = F.mse_loss(intermediate_output, target_data * -1)
            second_to_first_loss = F.mse_loss(intermediate_output, output)
            if any([torch.isnan(x) or torch.isinf(x) for x in [second_to_first_loss, first_to_target_loss, second_to_target_loss]]):
                print('is nan')
                is_diverged = True
                break

            loss = second_to_target_loss + first_to_target_loss
            df.loc[df.shape[0]] = [df.shape[0], loss.detach().numpy().item(),
                                   first_to_target_loss.detach().numpy().item(),
                                   second_to_target_loss.detach().numpy().item(),
                                   second_to_first_loss.detach().numpy().item()]
            loss.backward()
            optimizer.step()

        self.trained = True
        return df, is_diverged


if __name__ == '__main__':
    is_div = True
    while is_div:
        net = SecondaryNet(4, 2, 1, "SecondaryNet")
        data_df, is_div = net.self_train(20000, 25, 1e-4)
    from matplotlib import pyplot as plt
    import seaborn as sns
    # data_df = data_df[::-1]  # Reverse
    fig = sns.lineplot(data=data_df[[x for x in data_df.columns if x != 'step']])
    # fig.set(yscale='log')
    print(data_df.iloc[-1])
    print(data_df.iloc[0])
    plt.show()
    print("done")
