import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, Dense
from keras.layers import Input, TimeDistributed
from tqdm import tqdm

from typing import Union
import numpy as np


class Network(object):
    def __init__(self, features, cells, layers, bias=False, recurrent=False):
        self.features = features
        self.cells = cells
        self.num_layer = layers
        bias_params = cells if bias else 0

        # Recurrent network
        if recurrent:
            # First RNN
            p_layer_1 = (self.features * self.cells + self.cells ** 2 + bias_params)
            # All other RNN Layers
            p_layer_n = (self.cells * self.cells + self.cells ** 2 + bias_params) * (self.num_layer - 1)
        else:
            # First Dense
            p_layer_1 = (self.features * self.cells + bias_params)
            # All other Dense Layers
            p_layer_n = (self.cells * self.cells + bias_params) * (self.num_layer - 1)
        # Final Dense
        p_layer_out = self.features * self.cells + bias_params
        self.parameters = np.sum([p_layer_1, p_layer_n, p_layer_out])
        # Build network
        cell = SimpleRNN if recurrent else Dense
        self.inputs, x = Input(shape=(self.parameters // self.features, self.features,)), None

        for layer in range(self.num_layer):
            if recurrent:
                x = SimpleRNN(cells, activation=None, use_bias=False,
                              return_sequences=True)(self.inputs if layer == 0 else x)
            else:
                x = Dense(cells, activation=None, use_bias=False,
                              )(self.inputs if layer == 0 else x)
        self.outputs = Dense(self.features, activation=None, use_bias=False)(x)
        print('Network initialized, i haz {p} params @:{e}Features: {f}{e}Cells: {c}{e}Layers: {l}'.format(
            p=self.parameters, l=self.num_layer, c=self.cells, f=self.features, e='\n{}'.format(' ' * 5))
        )
        pass

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


class _BaseNetwork(Model):

    def __init__(self, **kwargs):
        super(_BaseNetwork, self).__init__(**kwargs)
        # This is dirty
        self.features = None

    def get_weights_flat(self):
        weights = super().get_weights()
        flat = np.asarray(np.concatenate([x.flatten() for x in weights]))
        return flat

    def step(self):
        flat = self.get_weights_flat()
        x = np.reshape(flat, (1, -1, self.features))
        return self.predict(x).flatten()

    def step_other(self, other: Union[Sequential, Model]) -> bool:
        pass

    def get_parameter_count(self):
        return np.sum([np.prod(x.shape) for x in self.get_weights()])

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError

    def compile(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def mean_abs_error(labels, predictions):
        return np.mean(np.abs(predictions - labels), axis=-1)

    @staticmethod
    def mean_sqrd_error(labels, predictions):
        return np.mean(np.square(predictions - labels), axis=-1)


class RecurrentNetwork(_BaseNetwork):
    def __init__(self, network: Network, *args, **kwargs):
        super().__init__(inputs=network.inputs, outputs=network.outputs)
        self.features = network.features
        self.parameters = network.parameters
        assert self.parameters == self.get_parameter_count()

    def fit(self, epochs=500, **kwargs):
        losses = []
        with tqdm(total=epochs, ascii=True,
                  desc='Type: {t}'. format(t=self.__class__.__name__),
                  postfix=["Loss", dict(value=0)]) as bar:
            for _ in range(epochs):
                y = self.step()
                weights = self.get_weights()
                global_idx = 0
                for idx, weight_matrix in enumerate(weights):
                    flattened = weight_matrix.flatten()
                    new_weights = y[global_idx:global_idx + flattened.shape[0]]
                    weights[idx] = np.reshape(new_weights, weight_matrix.shape)
                    global_idx += flattened.shape[0]
                losses.append(self.mean_sqrd_error(y.flatten(), self.get_weights_flat()))
                self.set_weights(weights)
                bar.postfix[1]["value"] = losses[-1]
                bar.update()
        return losses


class FeedForwardNetwork(_BaseNetwork):
    def __init__(self, network:Network, **kwargs):
        super().__init__(inputs=network.inputs, outputs=network.outputs, **kwargs)
        self.features = network.features
        self.parameters = network.parameters
        self.num_layer = network.num_layer
        assert self.parameters == self.get_parameter_count()

    def fit(self, epochs=500, **kwargs):
        losses = []
        with tqdm(total=epochs, ascii=True,
                  desc='Type: {t} @ Epoch:'. format(t=self.__class__.__name__),
                  postfix=["Loss", dict(value=0)]) as bar:
            for _ in range(epochs):
                y = self.step()
                weights = self.get_weights()
                # This is where i have to apply the aggregator
                global_idx = 0
                # This is where the weights are assigned to the new ones
                for idx, weight_matrix in enumerate(weights):
                    if self.num_layer == 1:
                        # In case of dense layers with a single layer, the RNN procedure can be applied
                        flattened = weight_matrix.flatten()
                    else:
                        # In case of multiple layers, a function aggregator has to be applied first.
                        # possible aggregators are: Mean, Transformation, Spektral analysis
                        pass
                    new_weights = y[global_idx:global_idx + flattened.shape[0]]
                    weights[idx] = np.reshape(new_weights, weight_matrix.shape)
                    global_idx += flattened.shape[0]
                losses.append(self.mean_sqrd_error(y.flatten(), self.get_weights_flat()))
                self.set_weights(weights)
                bar.postfix[1]["value"] = losses[-1]
                bar.update()
        return losses



if __name__ == '__main__':
    features, cells, layers = 2, 2, 2
    use_recurrent = False
    if use_recurrent:
        network = Network(features, cells, layers, recurrent=use_recurrent)
        r = RecurrentNetwork(network)
        loss = r.fit(epochs=10)
    else:
        network = Network(features, cells, layers, recurrent=use_recurrent)
        ff = FeedForwardNetwork(network)
        loss = ff.fit(epochs=10)
    print(loss)
