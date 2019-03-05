from experiment import *
from network import *
from soup import *
import numpy as np

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
