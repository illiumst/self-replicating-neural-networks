import copy
from typing import Dict, List
import torch
from tqdm import tqdm

from network import FixTypes, Net


def is_divergent(network: Net) -> bool:
    return network.input_weight_matrix().isinf().any().item() or network.input_weight_matrix().isnan().any().item()


def is_identity_function(network: Net, epsilon=pow(10, -5)) -> bool:

    input_data = network.input_weight_matrix()
    target_data = network.create_target_weights(input_data)
    predicted_values = network(input_data)

    return torch.allclose(target_data.detach(), predicted_values.detach(),
                          rtol=0, atol=epsilon)


def is_zero_fixpoint(network: Net, epsilon=pow(10, -5)) -> bool:
    target_data = network.create_target_weights(network.input_weight_matrix().detach())
    result = torch.allclose(target_data, torch.zeros_like(target_data), rtol=0, atol=epsilon)
    # result = bool(len(np.nonzero(network.create_target_weights(network.input_weight_matrix()))))
    return result


def is_secondary_fixpoint(network: Net, epsilon: float = pow(10, -5)) -> bool:
    """ Secondary fixpoint check is done like this: compare first INPUT with second OUTPUT.
    If they are within the boundaries, then is secondary fixpoint. """

    input_data = network.input_weight_matrix()
    target_data = network.create_target_weights(input_data)

    # Calculating first output
    first_output = network(input_data)

    # Getting the second output by initializing a new net with the weights of the original net.
    net_copy = copy.deepcopy(network)
    net_copy.apply_weights(first_output)
    input_data_2 = net_copy.input_weight_matrix()

    # Calculating second output
    second_output = network(input_data_2)

    # Perform the Check: all(epsilon > abs(input_data - second_output))
    check_abs_within_epsilon = torch.allclose(target_data.detach(), second_output.detach(),
                                              rtol=0, atol=epsilon)
    return check_abs_within_epsilon


def test_for_fixpoints(fixpoint_counter: Dict, nets: List, id_functions=None):
    id_functions = id_functions or list()

    for net in tqdm(nets, desc='Fixpoint Tester', total=len(nets)):
        if is_divergent(net):
            fixpoint_counter[FixTypes.divergent] += 1
            net.is_fixpoint = FixTypes.divergent
        elif is_identity_function(net):  # is default value
            fixpoint_counter[FixTypes.identity_func] += 1
            net.is_fixpoint = FixTypes.identity_func
            id_functions.append(net)
        elif is_zero_fixpoint(net):
            fixpoint_counter[FixTypes.fix_zero] += 1
            net.is_fixpoint = FixTypes.fix_zero
        elif is_secondary_fixpoint(net):
            fixpoint_counter[FixTypes.fix_sec] += 1
            net.is_fixpoint = FixTypes.fix_sec
        else:
            fixpoint_counter[FixTypes.other_func] += 1
            net.is_fixpoint = FixTypes.other_func
    return id_functions


def changing_rate(x_new, x_old):
    return x_new - x_old


def test_status(net: Net) -> Net:

    if is_divergent(net):
        net.is_fixpoint = FixTypes.divergent
    elif is_identity_function(net):  # is default value
        net.is_fixpoint = FixTypes.identity_func
    elif is_zero_fixpoint(net):
        net.is_fixpoint = FixTypes.fix_zero
    elif is_secondary_fixpoint(net):
        net.is_fixpoint = FixTypes.fix_sec
    else:
        net.is_fixpoint = FixTypes.other_func

    return net
