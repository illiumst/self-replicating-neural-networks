from experiment import *
from network import *
from soup import *
import numpy as np


class LearningNeuralNetwork(NeuralNetwork):

    @staticmethod
    def mean_reduction(weights, features):
        single_dim_weights = np.hstack([w.flatten() for w in weights])
        shaped_weights = np.reshape(single_dim_weights, (1, features, -1))
        x = np.mean(shaped_weights, axis=-1)
        return x

    @staticmethod
    def fft_reduction(weights, features):
        single_dim_weights = np.hstack([w.flatten() for w in weights])
        x = np.fft.fft(single_dim_weights, n=features)[None, ...]
        return x

    @staticmethod
    def random_reduction(_, features):
        x = np.random.rand(features)[None, ...]
        return x

    def __init__(self, width, depth, features, **kwargs):
        raise DeprecationWarning
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.features = features
        self.compile_params = dict(loss='mse', optimizer='sgd')
        self.model = Sequential()
        self.model.add(Dense(units=self.width, input_dim=self.features, **self.keras_params))
        for _ in range(self.depth - 1):
            self.model.add(Dense(units=self.width, **self.keras_params))
        self.model.add(Dense(units=self.features, **self.keras_params))
        self.model.compile(**self.compile_params)

    def apply_to_weights(self, old_weights, **kwargs):
        reduced = kwargs.get('reduction', self.fft_reduction)()
        raise NotImplementedError
        # build aggregations from old_weights
        weights = self.get_weights_flat()

        # call network
        old_aggregation = self.aggregate_fft(weights, self.aggregates)
        new_aggregation = self.apply(old_aggregation)

        # generate list of new weights
        new_weights_list = self.deaggregate_identically(new_aggregation, self.get_amount_of_weights())

        new_weights_list = self.get_shuffler()(new_weights_list)

        # write back new weights
        new_weights = self.fill_weights(old_weights, new_weights_list)

        # return results
        if self.params.get("print_all_weight_updates", False) and not self.is_silent():
            print("updated old weight aggregations " + str(old_aggregation))
            print("to new weight aggregations      " + str(new_aggregation))
            print("resulting in network weights ...")
            print(self.weights_to_string(new_weights))
        return new_weights

    def with_compile_params(self, **kwargs):
        self.compile_params.update(kwargs)
        return self

    def learn(self, epochs, reduction, batchsize=1):
        with tqdm(total=epochs, ascii=True,
                  desc='Type: {t} @ Epoch:'.format(t=self.__class__.__name__),
                  postfix=["Loss", dict(value=0)]) as bar:
            for epoch in range(epochs):
                old_weights = self.get_weights()
                x = reduction(old_weights, self.features)
                savestateCallback = SaveStateCallback(self, epoch=epoch)
                history = self.model.fit(x=x, y=x, verbose=0, batch_size=batchsize, callbacks=savestateCallback)
                bar.postfix[1]["value"] = history.history['loss'][-1]
                bar.update()


def vary(e=0.0, f=0.0):
    return [
        np.array([[1.0+e, 0.0+f], [0.0+f, 0.0+f], [0.0+f, 0.0+f], [0.0+f, 0.0+f]], dtype=np.float32),
        np.array([[1.0+e, 0.0+f], [0.0+f, 0.0+f]], dtype=np.float32),
        np.array([[1.0+e], [0.0+f]], dtype=np.float32)
    ]

if __name__ == '__main__':

    net = WeightwiseNeuralNetwork(width=2, depth=2).with_keras_params(activation='sigmoid')
    if False:
        net.set_weights([
            np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[1.0], [0.0]], dtype=np.float32)
        ])
        print(net.get_weights())
        net.self_attack(100)
        print(net.get_weights())
        print(net.is_fixpoint())

    if True:
        net.set_weights(vary(0.01, 0.0))
        print(net.get_weights())
        for _ in range(5):
            net.self_attack()
            print(net.get_weights())
        print(net.is_fixpoint())
