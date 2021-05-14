import copy
from typing import Dict, List
import numpy as np
from torch import Tensor
from network import Net


def overall_fixpoint_test(network: Net, epsilon: float, input_data) -> bool:
    predicted_values = network(input_data)

    check_smaller_epsilon = all(epsilon > predicted_values)
    check_greater_epsilon = all(-epsilon < predicted_values)

    if check_smaller_epsilon and check_greater_epsilon:
        return True
    else:
        return False


def is_divergent(network: Net) -> bool:
    for i in network.input_weight_matrix():
        weight_value = i[0].item()

        if np.isnan(weight_value) or np.isinf(weight_value):
            return True

    return False


def is_identity_function(network: Net, input_data: Tensor, target_data: Tensor, epsilon=pow(10, -5)) -> bool:
    predicted_values = network(input_data)

    return np.allclose(target_data.detach().numpy(), predicted_values.detach().numpy(), 0, epsilon)


def is_zero_fixpoint(network: Net, input_data: Tensor, epsilon=pow(10, -5)) -> bool:
    result = overall_fixpoint_test(network, epsilon, input_data)

    return result


def is_secondary_fixpoint(network: Net, input_data: Tensor, epsilon: float) -> bool:
    """ Secondary fixpoint check is done like this: compare first INPUT with second OUTPUT.
    If they are within the boundaries, then is secondary fixpoint. """

    # Calculating first output
    first_output = network(input_data)

    # Getting the second output by initializing a new net with the weights of the original net.
    net_copy = copy.deepcopy(network)
    net_copy.apply_weights(net_copy, first_output)
    input_data_2 = net_copy.input_weight_matrix()

    # Calculating second output
    second_output = network(input_data_2)

    check_smaller_epsilon = all(epsilon > second_output)
    check_greater_epsilon = all(-epsilon < second_output)

    if check_smaller_epsilon and check_greater_epsilon:
        return True
    else:
        return False


def is_weak_fixpoint(network: Net, input_data: Tensor, epsilon: float) -> bool:
    result = overall_fixpoint_test(network, epsilon, input_data)

    return result


def test_for_fixpoints(fixpoint_counter: Dict, nets: List, id_functions=[]):
    zero_epsilon = pow(10, -5)
    epsilon = pow(10, -3)

    for i in range(len(nets)):
        net = nets[i]
        input_data = net.input_weight_matrix()
        target_data = net.create_target_weights(input_data)

        if is_divergent(nets[i]):
            fixpoint_counter["divergent"] += 1
            nets[i].is_fixpoint = "divergent"
        elif is_identity_function(nets[i], input_data, target_data, zero_epsilon):
            fixpoint_counter["identity_func"] += 1
            nets[i].is_fixpoint = "identity_func"
            id_functions.append(nets[i])
        elif is_zero_fixpoint(nets[i], input_data, zero_epsilon):
            fixpoint_counter["fix_zero"] += 1
            nets[i].is_fixpoint = "fix_zero"
        elif is_weak_fixpoint(nets[i], input_data, epsilon):
            fixpoint_counter["fix_weak"] += 1
            nets[i].is_fixpoint = "fix_weak"
        elif is_secondary_fixpoint(nets[i], input_data, zero_epsilon):
            fixpoint_counter["fix_sec"] += 1
            nets[i].is_fixpoint = "fix_sec"
        else:
            fixpoint_counter["other_func"] += 1
            nets[i].is_fixpoint = "other_func"

    return id_functions

def changing_rate(x_new, x_old):
    return x_new - x_old
