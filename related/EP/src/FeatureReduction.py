import numpy as np
import numbers

class FeatureReduction():
    def __init__(self, type):
        self.type = type
        self.VecFromWeigths = None

    def calc(self, vec, n):
        self.weigthsToVec(vec)
        return {
            'fft' : self.fft(self.VecFromWeigths, n),
            'rfft': self.rfftn(self.VecFromWeigths, n),
            'mean': self.mean(self.VecFromWeigths,n),
            'meanShuffled':self.mean(self.shuffelVec(self.VecFromWeigths,3),n)
        }[self.type]

    def fft(self, vec, n):
        return np.fft.fft(vec, n)

    def rfftn(self, vec, n):
        return np.fft.rfft(vec, n)

    def shuffelVec(self, vec, mod):
        newVec = np.array([])
        rVec = np.array([])
        i = 0
        while i < len(vec):
            if i % mod == 0:
                newVec = np.append(newVec, vec[i])
            else:
                rVec = np.append(rVec, vec[i])
            i += 1
        if len(newVec) != len(vec):
            newVec = np.append(newVec, self.shuffelVec(rVec, mod))
        return newVec

    def mean(self,vec, n):
        '''
        Zerlegt einen Vektor in n gleich große Teile und berechnet den Mitteltwert.
        :param vec: Eingabevektor als array mit x Komponenten
        :param n: Die Größe des Ausgabevektors
        :return:Vektor als array mit n Komponenten
        '''
        if n > len(vec):
            Exception("n is bigger than len(vec) - no feature reduction avaiable")
        x = len(vec)/n
        result = np.array([])
        factor =1
        if x - int(x) != 0:
            factor = x - int(x)
        actFactor = factor
        vv = 0
        for value in vec:
                if round(x,5) <= 1:
                    x = len(vec) / n
                    vv += actFactor * value
                    result = np.append(result, [round(vv / (len(vec) / n),6)])
                    vv = (1 - actFactor) * value
                    if round((1 - actFactor),5) > 0:
                        x -= (1-actFactor)
                        actFactor += factor
                        if round(actFactor,5) > 1:
                            actFactor -= 1
                    else:
                        actFactor = factor
                else:
                    vv += value
                    x -= 1
        return result

    def weigthsToVec(self, weights, vec=np.array([])):
        '''
        Die Keras liefert die Gewichte eines neuronalen Netzwerkes in einem mehrdimensionalen Array. Dieses Array
        beinhaltet nicht nur die Gewichte der einzelnen Schichten, sondern auch der Status der Ausgabe der einzelnen
        Neuronen. Die Gewichte ines Netzes mit einem Neuron in der Eingabeschicht, zwei Neuronen in einer versteckten
        Schicht und einem Neuron in der Ausgabeschicht hat beispielsweise folgende Darstellung:
            [[[1 2]]
            [0. 0.]
            [[2] [3]]
            [0.]]
        Diese Funktion überführt die Darstellung in einen Vektor. Dabei werden die Informationen um die Ausgabe der
        einzelnen Neuronen verworfen. Der Vektor der die oben beschriebenen  Gewichte darstellt hat folgende Form:
        [1, 2, 2, 3]
        :param weights: mehrdimensionales Array der Gewichte aus Keras
        :return: Vektor in Form eines Arrays
        '''
        if isinstance(weights, np.float32):
            vec = np.append(vec, weights)
        else:
            for x in weights:
                if isinstance(x[0], np.ndarray):
                    for xx in x:
                        vec = np.append(vec, xx)
        self.VecFromWeigths = vec
