# from __future__ import annotations
import copy
import random
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from tqdm import tqdm


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


def prng():
    return random.random()


class FixTypes:

    divergent       = 'divergent'
    fix_zero        = 'fix_zero'
    identity_func   = 'identity_func'
    fix_sec         = 'fix_sec'
    other_func      = 'other_func'

    @classmethod
    def all_types(cls):
        return [val for key, val in cls.__dict__.items() if isinstance(val, str) and not key.startswith('_')]


class Net(nn.Module):

    @staticmethod
    def create_target_weights(input_weight_matrix: Tensor) -> Tensor:
        """ Outputting a tensor with the target weights. """

        # What kind of slow shit is this?
        # target_weight_matrix = np.arange(len(input_weight_matrix)).reshape(len(input_weight_matrix), 1).astype("f")
        # for i in range(len(input_weight_matrix)):
        #     target_weight_matrix[i] = input_weight_matrix[i][0]

        # Fast and simple
        return input_weight_matrix[:, 0].unsqueeze(-1)


    @staticmethod
    def are_weights_diverged(network_weights):
        """ Testing if the weights are eiter converging to infinity or -infinity. """

        # Slow and shitty:
        # for layer_id, layer in enumerate(network_weights):
        #     for cell_id, cell in enumerate(layer):
        #         for weight_id, weight in enumerate(cell):
        #             if torch.isnan(weight):
        #                 return True
        #             if torch.isinf(weight):
        #                 return True
        # return False
        # Fast and modern:
        return any(x.isnan.any() or x.isinf().any() for x in network_weights.parameters)

    def apply_weights(self, new_weights: Tensor):
        """ Changing the weights of a network to new given values. """
        with torch.no_grad():
            i = 0
            for parameters in self.parameters():
                size = parameters.numel()
                parameters[:] = new_weights[i:i+size].view(parameters.shape)[:]
                i += size
        return self

    def __init__(self, i_size: int, h_size: int, o_size: int, name=None, start_time=1) -> None:
        super().__init__()
        self.start_time = start_time

        self.name = name
        self.child_nets = []

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

        self.is_fixpoint = FixTypes.other_func
        self.layers = nn.ModuleList(
            [nn.Linear(i_size, h_size, False),
             nn.Linear(h_size, h_size, False),
             nn.Linear(h_size, o_size, False)]
        )

        self._weight_pos_enc_and_mask = None
        self.apply(xavier_init)

    @property
    def _weight_pos_enc(self):
        if self._weight_pos_enc_and_mask is None:
            d = next(self.parameters()).device
            weight_matrix = []
            for layer_id, layer in enumerate(self.layers):
                x = next(layer.parameters())
                weight_matrix.append(
                    torch.cat(
                        (
                            # Those are the weights
                            torch.full((x.numel(), 1), 0, device=d),
                            # Layer enumeration
                            torch.full((x.numel(), 1), layer_id, device=d),
                            # Cell Enumeration
                            torch.arange(layer.out_features, device=d).repeat_interleave(layer.in_features).view(-1, 1),
                            # Weight Enumeration within the Cells
                            torch.arange(layer.in_features, device=d).view(-1, 1).repeat(layer.out_features, 1),
                            *(torch.full((x.numel(), 1), 0, device=d) for _ in range(self.input_size-4))
                        ), dim=1)
                )
            # Finalize
            weight_matrix = torch.cat(weight_matrix).float()

            # Normalize 1,2,3 column of dim 1
            last_pos_idx = self.input_size - 4
            max_per_col, _ = weight_matrix[:, 1:-last_pos_idx].max(keepdim=True, dim=0)
            weight_matrix[:, 1:-last_pos_idx] = (weight_matrix[:, 1:-last_pos_idx] / max_per_col) + 1e-8

            # computations
            # create a mask where pos is 0 if it is to be replaced
            mask = torch.ones_like(weight_matrix)
            mask[:, 0] = 0

            self._weight_pos_enc_and_mask = weight_matrix, mask
        return tuple(x.clone() for x in self._weight_pos_enc_and_mask)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def normalize(self, value, norm):
        raise NotImplementedError
        # FIXME, This is bullshit, the code does not do what the docstring explains
        # Obsolete now
        """ Normalizing the values >= 1 and adding pow(10, -8) to the values equal to 0 """

        if norm > 1:
            return float(value) / float(norm)
        else:
            return float(value)

    def input_weight_matrix(self) -> Tensor:
        """ Calculating the input tensor formed from the weights of the net """
        weight_matrix = torch.cat([x.view(-1, 1) for x in self.parameters()])
        pos_enc, mask = self._weight_pos_enc
        weight_matrix = pos_enc * mask + weight_matrix.expand(-1, pos_enc.shape[-1]) * (1 - mask)
        return weight_matrix

    def target_weight_matrix(self) -> Tensor:
        weight_matrix = torch.cat([x.view(-1, 1) for x in self.parameters()])
        return weight_matrix

    def self_train(self,
                   training_steps: int,
                   log_step_size: int = 0,
                   learning_rate: float = 0.0004,
                   save_history: bool = True
                   ) -> (Tensor, list):
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

            if save_history:
                # Saving the history of the weights after a certain amount of steps (aka log_step_size) for research.
                # If it is a soup/mixed env. save weights only at the end of all training steps (aka a soup/mixed epoch)
                if "soup" not in self.name and "mixed" not in self.name:
                    weights = self.create_target_weights(self.input_weight_matrix())
                    # If self-training steps are lower than 10, then append weight history after each ST step.
                    if self.number_trained < 10:
                        self.s_train_weights_history.append(weights.T.detach().numpy())
                        self.loss_history.append(loss.item())
                    else:
                        if log_step_size != 0:
                            if self.number_trained % log_step_size == 0:
                                self.s_train_weights_history.append(weights.T.detach().numpy())
                                self.loss_history.append(loss.item())

        weights = self.create_target_weights(self.input_weight_matrix())
        # Saving weights only at the end of a soup/mixed exp. epoch.
        if save_history:
            if "soup" in self.name or "mixed" in self.name:
                self.s_train_weights_history.append(weights.T.detach().numpy())
                self.loss_history.append(loss.item())

        self.trained = True
        return loss, self.loss_history

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

    def self_train(self, training_steps: int, log_step_size: int, learning_rate: float) -> (pd.DataFrame, Tensor, list):
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


class MetaCell(nn.Module):
    def __init__(self, name, interface, weight_interface=5, weight_hidden_size=2, weight_output_size=1):
        super().__init__()
        self.name = name
        self.interface = interface
        self.weight_interface = weight_interface
        self.net_hidden_size = weight_hidden_size
        self.net_ouput_size = weight_output_size
        self.meta_weight_list = nn.ModuleList(
            [Net(self.weight_interface, self.net_hidden_size,
                 self.net_ouput_size, name=f'{self.name}_W{weight_idx}'
                 ) for weight_idx in range(self.interface)]
        )
        self.__bed_mask = None

    @property
    def _bed_mask(self):
        if self.__bed_mask is None:
            d = next(self.parameters()).device
            embedding = torch.zeros(1, self.weight_interface, device=d)

            # computations
            # create a mask where pos is 0 if it is to be replaced
            mask = torch.ones_like(embedding)
            mask[:, -1] = 0

            self.__bed_mask = embedding, mask
        return tuple(x.clone() for x in self.__bed_mask)

    def forward(self, x):
        embedding, mask = self._bed_mask
        expanded_mask = mask.expand(*x.shape, embedding.shape[-1])
        embedding = embedding.repeat(*x.shape, 1)

        # Row-wise
        # xs = x.unsqueeze(-1).expand(-1, -1, embedding.shape[-1]).swapdims(0, 1)
        # Column-wise
        xs = x.unsqueeze(-1).expand(-1, -1, embedding.shape[-1])
        xs = embedding * expanded_mask + xs * (1 - expanded_mask)
        # ToDo Speed this up!
        tensor = torch.hstack([meta_weight(xs[:, idx, :]) for idx, meta_weight in enumerate(self.meta_weight_list)])

        tensor = torch.sum(tensor, dim=-1, keepdim=True)
        return tensor

    @property
    def particles(self):
        return (net for net in self.meta_weight_list)


class MetaLayer(nn.Module):
    def __init__(self, name, interface=4, width=4,  # residual_skip=False,
                 weight_interface=5, weight_hidden_size=2, weight_output_size=1):
        super().__init__()
        self.residual_skip = False
        self.name = name
        self.interface = interface
        self.width = width

        self.meta_cell_list = nn.ModuleList([
            MetaCell(name=f'{self.name}_C{cell_idx}',
                     interface=interface,
                     weight_interface=weight_interface, weight_hidden_size=weight_hidden_size,
                     weight_output_size=weight_output_size,
                     ) for cell_idx in range(self.width)]
        )

    def forward(self, x):
        cell_results = []
        for metacell in self.meta_cell_list:
            cell_results.append(metacell(x))
        tensor = torch.hstack(cell_results)
        if self.residual_skip and x.shape == tensor.shape:
            tensor += x
        return tensor

    @property
    def particles(self):
        return (weight for metacell in self.meta_cell_list for weight in metacell.particles)


class MetaNet(nn.Module):

    def __init__(self, interface=4, depth=3, width=4, out=1, activation=None, residual_skip=True, dropout=0,
                 weight_interface=5, weight_hidden_size=2, weight_output_size=1,):
        super().__init__()
        self.residual_skip = residual_skip
        self.dropout = dropout
        self.activation = activation
        self.out = out
        self.interface = interface
        self.width = width
        self.depth = depth
        self.weight_interface = weight_interface
        self.weight_hidden_size = weight_hidden_size
        self.weight_output_size = weight_output_size
        self._meta_layer_first = MetaLayer(name=f'L{0}',
                                           interface=self.interface,
                                           width=self.width,
                                           weight_interface=weight_interface,
                                           weight_hidden_size=weight_hidden_size,
                                           weight_output_size=weight_output_size)

        self._meta_layer_list = nn.ModuleList([MetaLayer(name=f'L{layer_idx + 1}',
                                                         interface=self.width, width=self.width,
                                                         weight_interface=weight_interface,
                                                         weight_hidden_size=weight_hidden_size,
                                                         weight_output_size=weight_output_size,

                                                         ) for layer_idx in range(self.depth - 2)]
                                              )
        self._meta_layer_last = MetaLayer(name=f'L{len(self._meta_layer_list)}',
                                          interface=self.width, width=self.out,
                                          weight_interface=weight_interface,
                                          weight_hidden_size=weight_hidden_size,
                                          weight_output_size=weight_output_size,
                                          )
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self._all_layers_with_particles = [self._meta_layer_first, *self._meta_layer_list, self._meta_layer_last]

    def replace_with_zero(self, ident_key):
        replaced_particles = 0
        for particle in self.particles:
            if particle.is_fixpoint == ident_key:
                particle.load_state_dict(
                    {key: torch.zeros_like(state) for key, state in particle.state_dict().items()}
                )
                replaced_particles += 1
        tqdm.write(f'Particle Parameters replaced: {str(replaced_particles)}')
        return self

    def forward(self, x):
        if self.dropout != 0:
            x = self.dropout_layer(x)
        tensor = self._meta_layer_first(x)
        for idx, meta_layer in enumerate(self._meta_layer_list, start=1):
            if self.dropout != 0:
                tensor = self.dropout_layer(tensor)
            if idx % 2 == 1 and self.residual_skip:
                x = tensor.clone()
            tensor = meta_layer(tensor)
            if idx % 2 == 0 and self.residual_skip:
                tensor = tensor + x
        if self.dropout != 0:
            x = self.dropout_layer(x)
        tensor = self._meta_layer_last(x)
        return tensor

    @property
    def particles(self):
        return (cell for metalayer in self._all_layers_with_particles for cell in metalayer.particles)

    def combined_self_train(self):
        losses = []
        for particle in self.particles:
            # Intergrate optimizer and backward function
            input_data = particle.input_weight_matrix()
            target_data = particle.create_target_weights(input_data)
            output = particle(input_data)
            losses.append(F.mse_loss(output, target_data))
        return torch.hstack(losses).sum(dim=-1, keepdim=True)

    @property
    def hyperparams(self):
        return {key: val for key, val in self.__dict__.items() if not key.startswith('_')}

    def replace_particles(self, particle_weights_list):
        for layer in self._all_layers_with_particles:
            for cell in layer.meta_cell_list:
                # Individual replacement on cell lvl
                for weight in cell.meta_weight_list:
                    weight.apply_weights(next(particle_weights_list).detach())
        return self


class MetaNetCompareBaseline(nn.Module):

    def __init__(self, interface=4, depth=3, width=4, out=1, activation=None):
        super().__init__()
        self.activation = activation
        self.out = out
        self.interface = interface
        self.width = width
        self.depth = depth

        self._meta_layer_list = nn.ModuleList()

        self._meta_layer_list.append(nn.Linear(self.interface, self.width, bias=False))
        self._meta_layer_list.extend([nn.Linear(self.width, self.width, bias=False) for _ in range(self.depth - 2)])
        self._meta_layer_list.append(nn.Linear(self.width, self.out, bias=False))

    def forward(self, x):
        tensor = x
        for meta_layer in self._meta_layer_list:
            tensor = meta_layer(tensor)
        return tensor


if __name__ == '__main__':
    metanet = MetaNet(interface=3, depth=5, width=3, out=1, residual_skip=True)
    next(metanet.particles).input_weight_matrix()
    metanet(torch.hstack([torch.full((2, 1), 1.0) for _ in range(metanet.interface)]))
    a = metanet.particles
    print('Test')
    print('Test')
    print('Test')
    print('Test')
    print('Test')
