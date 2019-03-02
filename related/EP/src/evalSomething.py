import matplotlib
matplotlib.use('Agg')
import datetime
from keras.utils import plot_model

import copy
import numpy as np
import decimal

try:
    from src.PltData import PltData
    from src.Functions import Functions
    from src.NeuralNetwork import NeuralNetwork
    from src.FeatureReduction import FeatureReduction
except ImportError:
    from PltData import PltData
    from NeuralNetwork import NeuralNetwork
    from FeatureReduction import FeatureReduction
    import Functions

def getRangeAroundNumber(myNumber, rangeWidth):
    '''
    Gibt einen Zahlen Bereich rund um myNumber aus
    :param myNumber:
    :return:
    '''
    try:
        myNumber = myNumber.real
    except TypeError:
        myNumber = myNumber
    d = decimal.Decimal(myNumber.real)
    ex = 3
    print(myNumber*(10**ex))
    start = myNumber*(10**ex)-rangeWidth
    stop = myNumber * (10**ex) + rangeWidth
    data = np.arange(start, stop, 1)
    data = data / (10**ex)
    return data

def evalSomething(numberOfNeurons, activationFunctions, featureReduction,
                 numberLoops, loss):

    nn = NeuralNetwork(numberOfNeurons, activationFunctions, featureReduction,
                       numberLoops, loss, printVectors=False)
    nn.addLayers()
    nn.loadModel()
    weights = nn.model.get_weights()
    data = np.array([nn.featureReductionFunction.calc(weights, nn.numberOfNeurons[0])])
    fp = data[0][0]
    #data = getRangeAroundNumber(fp, 30)
    data = np.arange(-10000, 10000, 1)
    start = min(data)
    stop = max(data)
    step = abs((start-stop)/len(data))
    text = nn.getDescription() +"\nFixpunkt: "+ str(fp)
    nn.evaluate(data, str(start)+"_"+str(stop)+"_"+str(step), text= text + "\nStart: " +str(start) +"\nStop "+ str(stop) + "\nStep "+ str(step))

v = np.array([1,2,3,24])
for i in v:
    evalSomething(numberOfNeurons=[1, i, 1], activationFunctions=["sigmoid", "linear"], featureReduction='rfft',
                 numberLoops=100000, loss='mean_squared_error')
    i-=1



