import glob
import random
import numpy as np

def checkFileExists(fileName):
    '''
    prüft ob eine Datei existiert und gibt gegebenenfalls den Pfad zurück, sonst False
    :param fileName:
    :return: Boolean
    '''

    file = glob.glob(fileName)
    if len(file) > 0:
        return file[0]
    else:
        return False

def calcMeanSquaredError(a, b):
    '''
    Berechnet den MSE zwischen a und b
    :param a:
    :param b:
    :return: MSE
    '''

    a = a.astype(float)
    b = b.astype(float)
    mse = ((a - b) ** 2).mean()
    return mse

def calcScale(array):
    '''
    Berechnet die Skala eines Arrays
    :param array:
    :return:
    '''
    return abs(max(array)-min(array))

def getRandomLayer(tuple, standardDeviation=0.01):
    '''
    Liefert ein zufällige Gewichte für die übergebene Schicht im Keras Format
    :param tuple: ein Tupel für die Schicht (12, 75) ist beispielweise eine Schicht mit 12 auf 75 Neuronen
    :param standardDeviation:
    :return: Zufällige Gewichte für die Schicht im Keras Format
    '''
    randomLayer = []
    i = 0
    while i < tuple[0]:
        randomNeuron = []
        x = 0
        while x < tuple[1]:
            randomNeuron.append(getRandomGausNumber(standardDeviation))
            x += 1
        randomLayer.append(randomNeuron)
        i+= 1
    randomLayer = [randomLayer]
    randomLayer.append([0.]*tuple[1])
    return randomLayer

def getRandomGausNumber(standardDeviation):
    '''
    Liefert eine Zufallszahl um null mit der angegebenen Standardabweichung
    :param standardDeviation:
    :return:
    '''
    return np.random.normal(0.0, standardDeviation)