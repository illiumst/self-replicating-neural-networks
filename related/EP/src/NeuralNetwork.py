import matplotlib
matplotlib.use('Agg')
import datetime
from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adadelta
from keras.models import load_model
from keras import utils
import copy
import numpy as np
import collections
from operator import add

try:
    from src.Functions import Functions
    from src.PltData import PltData
    from src.FeatureReduction import FeatureReduction
    from src.LossHistory import LossHistory
except ImportError:
    import Functions
    from PltData import PltData
    from FeatureReduction import FeatureReduction
    from LossHistory import LossHistory

class NeuralNetwork:

    def __init__(self, numberOfNeurons, activationFunctions, featureReduction,
                 numberLoops, loss='mean_squared_error', printVectors=False, path="../images/",
                 fitByHillClimber=False, standardDeviation = 0.01, numberOtRandomShots=20, checkNewWeightsIsReallyBetter=False):
        '''
        :param numberOfNeurons: Array mit Integers. Gibt die Anzahl der Neuronen der einzelnen Schichten an
        :param activationFunctions: Array mit Strings für die Aktivierungsfunktionen der einzelnen Schichten
        :param featureReduction: String der die Funktion zur Feature Reduzierung angibt
        :param numberLoops: die Anzahl der Schleifendruchläufe
        :param loss: Fehlerfunktion
        :param printVectors: Boolean gibt an ob die Gewichte und der dazugehörige die die feature reduction funktion
        transformierte Vektor ausgeben werden soll
        :param path: der Pfad zu den Ergebnissen
        '''
        backend.clear_session()
        self.model = Sequential()
        self.optimzier = Adadelta()
        self.epochs = 1
        self.fitByHillClimber = fitByHillClimber
        self.checkNewWeightsIsReallyBetter=checkNewWeightsIsReallyBetter
        self.numberOtRandomShots = numberOtRandomShots
        self.standardDeviation = standardDeviation
        self.numberOfNeurons = numberOfNeurons
        self.activationFunctions = activationFunctions
        self.featureReduction = featureReduction
        self.featureReductionFunction = FeatureReduction(self.featureReduction)
        self.numberLoops = numberLoops
        self.loss = loss
        self.addedLayers = "No_Layers_added"
        self.result = []
        self.fileForVectors = None
        self.printVectors = printVectors
        self.path = path
        self.beginGrowing = 0
        self.stopGrowing = 0
        self.LM = 0
        self.dataHistory = []
        self.minFailure = 100
        self.minFailureLoop = None

    def addLayers(self):
        i = 2
        self.addedLayers = "inputDim_" + str(self.numberOfNeurons[0]) + "_" + str(self.activationFunctions[0]) + \
                           "_" + str(self.numberOfNeurons[1])
        self.model.add(Dense(self.numberOfNeurons[1], kernel_initializer="uniform", input_dim=self.numberOfNeurons[0],
                             activation=self.activationFunctions[0]))
        while i < len(self.numberOfNeurons):
            self.addLayer(self.activationFunctions[i-1], self.numberOfNeurons[i])
            i+= 1

    def addLayer(self, activationFunction, numberOfNeurons):
        self.model.add(Dense(numberOfNeurons, kernel_initializer="uniform"))
        self.model.add((Activation(activationFunction)))
        self.addedLayers += "_" + activationFunction+"_"+ str(numberOfNeurons)

    def fitByStochasticHillClimberV3(self, inputD, outputD, callbacks=None):
        '''
        Diese Version des stochastischen Hill Climbers, überorüft den Fehler nur Anhand der neuen Gewichte.
        :param inputD:
        :param outputD:
        :param callbacks:
        :return:
        '''

        weights = self.model.get_weights()
        aktWeights = self.model.get_weights()
        memDict = {}
        i = 0
        if callbacks != None:
            for f in callbacks:
                f.on_train_begin()
        while i <= self.numberOtRandomShots:
            i+= 1
            loss = Functions.calcMeanSquaredError(self.model.predict(inputD, batch_size=1), outputD)
            if i == 1:
                if callbacks != None:
                    for f in callbacks:
                        f.addLoss(loss)
            memDict[loss] = weights
            weights = self.joinWeights(self.getRandomWeights(),weights)
            self.model.set_weights(weights)
            inputD = np.array([self.featureReductionFunction.calc(self.model.get_weights(), self.numberOfNeurons[0])])
            outputD = inputD

        od = collections.OrderedDict(sorted(memDict.items()))
        od = list(od.items())
        self.model.set_weights(od[0][1])
        return

    def fitByStochasticHillClimber(self, inputD, outputD, callbacks=None):
        '''
        Die ersten beiden Versionen des Hill Climber.
        V1 wird ausgeführt wenn self.checkNewWeightsIsReallyBetter nicht True ist. In diesem Fall wird nur gegen die
        alten Gewichte und dessen Repräsentation geprüft.
        V2 wird ausgeführt wenn self.checkNewWeightsIsReallyBetter True ist. In diesem Fall wird ein zweiter Check auf
        die neuen Gewichte ausgeführt. Nur wenn beide ein besseres Ergebnis liefern, werden die neuen Gewichte übernommen.
        Die bessere Variante ist fitByStochasticHillClimberV3 - in der nur gegen die neuen Gewichte geprüft wird.
        :param inputD:
        :param outputD:
        :param callbacks:
        :return:
        '''
        weights = self.model.get_weights()
        aktWeights = self.model.get_weights()
        memDict = {}
        i = 0
        if callbacks != None:
            for f in callbacks:
                f.on_train_begin()
        while i <= self.numberOtRandomShots:
            i+= 1
            loss = Functions.calcMeanSquaredError(self.model.predict(inputD), outputD)
            if i == 1:
                if callbacks != None:
                    for f in callbacks:
                        f.addLoss(loss)
            memDict[loss] = weights
            weights = self.joinWeights(self.getRandomWeights(),weights)
            self.model.set_weights(weights)
        od = collections.OrderedDict(sorted(memDict.items()))
        od = list(od.items())
        if self.checkNewWeightsIsReallyBetter:
            self.model.set_weights(od[0][1])
            iData = np.array([self.featureReductionFunction.calc(self.model.get_weights(), self.numberOfNeurons[0])])
            errorWithNewWeights = Functions.calcMeanSquaredError(self.model.predict(iData, batch_size=1), iData)
            self.model.set_weights(aktWeights)
            errorWithOldWeights = Functions.calcMeanSquaredError(self.model.predict(iData, batch_size=1), iData)
            if errorWithNewWeights <errorWithOldWeights:
                self.model.set_weights(od[0][1])
        else:
            self.model.set_weights(od[0][1])
        #print(Functions.calcMeanSquaredError(self.model.predict(inputD), outputD), od[0][0])
        return

    def removeAFOutputFromWeightsArray(self,weights):
        '''
        der Output von model.get_weigths() liefert nicht nur die Gewichte des Netzes sondern auch die aktuelle Ausgabe,
        der Neuronen. Manchmal ist es nötig für weitere Berechungen diese Ausgabe zu entfernen.
        :param weights:
        :return:
        '''
        newWeights = []
        for value in weights:
            if isinstance(value[0], list) or isinstance(value[0], np.ndarray):
                newWeights.append(value)
        return newWeights

    def joinWeights(self, first, second):
        '''
        addiert zwei Arrays die die Gewichte darstellen.
        :param first:
        :param second:
        :return:
        '''
        newWeights = copy.deepcopy(first)
        x = 0
        for myList in first:
            if isinstance(myList[0], list):
                #gewichte addieren
                newWeights[x] = self.joinArrays(myList,second[x])
            x += 1
        return newWeights

    def joinArrays(self, first, second):
        x = 0
        for value in first:
            if isinstance(value, list):
                first[x] = self.joinArrays(first[x], second[x])
            else:
                first[x] += second[x]
            x+= 1
        return first

    def getRandomWeights(self):
        '''
        liefert zufällig generierte Gewichte für die aktuelle Netzkonfiguration
        :return:
        '''
        i = 0
        while i+1 < len(self.numberOfNeurons):
            tuple = (self.numberOfNeurons[i],self.numberOfNeurons[i+1])
            layer = Functions.getRandomLayer(tuple)
            if i == 0:
                weights= layer
            else:
                for list in layer:
                    weights.append(list)
            i+=1
        return weights


    def fit(self, stepWise=False, checkLM=False, searchForThreshold=False, checkScale=False):
        numberLoops  = self.numberLoops
        self.model.compile(loss=self.loss, optimizer=self.optimzier)
        history = LossHistory()
        i = 0
        iamHere = False
        while i < numberLoops:
            weights = self.model.get_weights()
            data = np.array([self.featureReductionFunction.calc(weights, self.numberOfNeurons[0])])
            self.dataHistory.append(data)
            if self.printVectors:
                self.printVec(i, self.featureReductionFunction.VecFromWeigths, data)
            if not self.fitByHillClimber:
                self.model.fit(data, data, epochs=1, callbacks=[history], verbose=0)
            else:
                self.fitByStochasticHillClimberV3(data, data, callbacks=[history])
                if history.losses[-1] < self.minFailure:
                    self.minFailure = history.losses[-1]
                    self.minFailureLoop = i
            self.result.append(history.losses[-1])

            i += 1
            if checkScale:
                dd = np.sum(np.array(self.result[-1000:]))
                if self.checkGrowing(self.result, 10) or dd == 0. or i > 2500:
                    break
            if searchForThreshold:
                if self.checkGrowing(self.result, 100):
                    return self.result[0], True
                if i > 1000:
                    return self.result[0], False
            if checkLM:
                if len(self.result) > 1000:
                    dd = np.sum(np.array(self.result[-1000:]))
                    if dd == 0.:
                        # Wenn die Summe der letzten 1000 Fehler echt null ist - muss ein Fixpunkt gefunden worden sein
                        self.beginGrowing = 0
                        break
                if self.checkGrowing(self.result, 10) and self.beginGrowing == 0:
                # Fehler steigt wieder
                    self.beginGrowing = i
                    if stepWise:
                        self.numberLoops = i
                        self.printEvaluation()
                if self.beginGrowing > 0:
                    '''
                    if len(self.result) > 1000:
                        if (i > 10000 and self.checkGrowing(self.result, 100)):
                            if stepWise:
                                self.numberLoops = i
                                self.printEvaluation()
                            #print("BREAK 1", round(dd,6), dd)
                            break
                    '''
                    if not self.checkGrowing(self.result, 10, checkSame=False) and i-self.beginGrowing>500 and not iamHere:
                    # In einigen Fällen ist der Wachstum sehr langsam, deswegen checkSame = False,
                    # nach beginGrowing kommt es manchmal vor, dass der Wachstum kurz aussetzt, deswegen
                    # müssen sollten zwischen beginGrowing und endGrowing 500 Schritte liegen
                        self.stopGrowing = i
                        self.LM =self.result[len(self.result) - 1]
                        if stepWise:
                            self.numberLoops = i
                            self.printEvaluation()
                            iamHere = True
                        else:
                            break

        pl = PltData(np.array(self.result))
        pl.linePlot(self.getFileName(i), width=1600, text=self.getDescription())

    def printEvaluation(self):
        start = -100000
        stop = 100000
        step = 1
        data = np.arange(start, stop, step)
        self.evaluate(data, str(start) + "_" + str(stop) + "_" + str(step),
                    text=self.getDescription() + "\nStart: " + str(start) + "\nStop " + str(stop) + "\nStep " + str(step))

    def checkGrowing(self, mArray, range, checkSame=True):
        if len(mArray) < range*2:
            return False
        values = np.array(mArray[-1*range*2:])
        values = values.reshape(-1, int(len(values)/2))
        if np.sum(values[0]) == np.sum(values[1]) and checkSame:
            return False
        if np.sum(values[0]) > np.sum(values[1]):
            return False
        else:
            return True

    def evaluate(self, inputData, filename, text=""):
        pD = self.model.predict(inputData, batch_size=1)
        pD = np.reshape(pD,(1,len(pD)))[0]
        pl = PltData(pD)
        pl.linePlot(self.path + self.getFileName() + "_" + filename, width=1024, text=text, xLegend="X", yLegend="Y", x=inputData, yTextPos=-0.0015, xTextPos=-10000)

    def loadModel(self):
        file = "../nnModels/" + self.getFileName() + ".h5"
        if not Functions.checkFileExists(file):
            print("no model found in " + str(file))
            return
        self.model = load_model(file)
        print("Model loaded from " + str(file))

    def saveModel(self):
        self.model.save("../nnModels/" + str(self.getFileName()) + ".h5")

    def printVec(self, loop, weight, input):
        if self.fileForVectors == None:
            self.fileForVectors = open(self.path + self.getFileName() + ".txt", "w")
            self.fileForVectors.write("numberOfLoop \t weights \t " +self.featureReduction + " result\n")
        self.fileForVectors.write(str(loop) + " \t " + repr(weight) +" \t " + repr(input) + " \n")


    def getFileName(self, numberLoops=None):
        if numberLoops is None:
            numberLoops = self.numberLoops
        fileName ="nOL_" + str(len(self.activationFunctions)) + \
                 self.addedLayers + "_nLoops_" + str(numberLoops) + "_fR_" +self.featureReduction
        if self.fitByHillClimber:
            fileName += "_standardDeviation_" +str(self.standardDeviation) +\
                        "_numberOtRandomShots_"+str(self.numberOtRandomShots)
            if self.checkNewWeightsIsReallyBetter:
                        fileName += "_checkNewWeightsIsReallyBetter"
        return fileName

    def getDescription(self):
        text = "Loops: " + str(self.numberLoops) + \
               "\nLayers:" + self.getLayerText() +\
               "\nOptimizer: Adadelta" + \
               "\nFeature Reduction: " + self.featureReduction
        if self.fitByHillClimber:
            text += "\nStandard Deviation: " +str(self.standardDeviation) +\
                    "\nNumber Of Random Shots: " + str(self.numberOtRandomShots) +\
                    "\nMinimaler Fehler: " +str(self.minFailure) +" bei Loop: " + str(self.minFailureLoop)+\
                    "\ncheckNewWeightsIsReallyBetter:" + str(self.checkNewWeightsIsReallyBetter)

        return text

    def getLayerText(self):
        text = ""
        i = 0
        for l in self.activationFunctions:
            text += "\n" + str(i+1) + " Layer " + l + " " + str(self.numberOfNeurons[i+1]) + " Neuronen"
            i += 1
        return text

