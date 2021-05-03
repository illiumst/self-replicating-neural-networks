import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class PltData:

    def __init__(self, data):
        self.data = data


    def linePlot(self, fileName, x = None, legend=[], text="", yLegend="Mittle quadratische Fehler", xLegend="Anzahl der Loops",plotText = True, dest="images",
                 height=800, multiFigs=False, pltDiff=True, width=None, xTextPos = 0, yTextPos = 0):
        myDbi = 96
        dim = len(self.data.shape)
        if width == None:
            try:
                size = len(self.data[0]) * 4.5 / myDbi
            except TypeError:
                size = len(self.data)* 4.5 /myDbi
        else:
            size = width/myDbi
        if size > 2^16:
            size = 2^16

        height = height/myDbi
        plt.figure(figsize=(size, height))

        if dim == 2:
            i = 0
            for row in self.data:
                if len(legend) > 0:
                    label = legend[i]
                else:
                    label = ""
                if multiFigs:
                    if i == 0:
                        f, axarr = plt.subplots(len(self.data), sharex=True, sharey=True)
                        f.set_figheight(height)
                        f.set_figwidth(size)
                        f.text(0.04, 0.5, yLegend, va='center', rotation='vertical')
                    axarr[i].plot(row)
                    axarr[i].grid(True)
                    axarr[i].set_ylabel(label)

                else:
                    if x is None:
                        plt.plot(row, label=label)
                    else:
                        plt.plot(x, row, label=label)
                i += 1
            if pltDiff:
                plt.plot(np.subtract(self.data[0].astype(float), self.data[1].astype(float)), label="Differenz")
        else:
            if x is None:
                plt.plot(self.data)
            else:
                plt.plot(x,self.data)

        plt.legend()
        if not multiFigs:
            plt.ylabel(yLegend)
        plt.xlabel(xLegend)
        if plotText:
            plt.text(0+xTextPos, np.amax(self.data)+yTextPos, text)
        plt.grid(True)
        plt.savefig("../"+ dest + "/" + fileName+ ".png", bbox_inches='tight')
        plt.close()

    def plotNNModel(self, data, fileName, dest="images"):
        data, pos = self.getModelData(data)
        Gp = nx.Graph()
        Gp.add_weighted_edges_from(data)
        plt.figure(figsize=(15, 15))
        nx.draw(Gp, pos=pos)
        nx.draw_networkx_labels(Gp, pos=pos)
        nx.draw_networkx_edge_labels(Gp, pos=pos)
        plt.savefig("../" + dest + "/" + fileName, bbox_inches='tight')
        plt.close()

    def plotPoints(self, data, labels, filename, xlabel = ""):
        plt.figure(figsize=(1600/96, 5))
        i = 0
        dots = ["ro", "go", "bo","yo"]
        for row in data:
            plt.plot(row, [i] * (len(row)), dots[i], label=labels[i])
            i+=1
        plt.legend()
        plt.xlabel(xlabel)
        plt.grid(True)

        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def getModelData(self, data):
        pos = {}
        xRange, yRange = self.getPositionRanges(data)
        layer = 1000
        layerSteps = 1000
        modelData = []
        firstLayer = True
        z = 0
        r = 0
        while z < len(data)-1:
            x = data[z]
            if firstLayer: nextNodeNumber = 0
            nodeNumber = 0
            if isinstance(x[0], np.ndarray):
                xPos = xRange[r]
                xPosNext = xRange[r+1]
                r += 1
                if firstLayer:
                    yKor = int(self.getLenOfFirstLayer(x)% len(yRange)/2)
                else:
                    yKor = int((len(data[z+1])% len(yRange))/2)
                if len(data) > z+3:
                    yKorNext = int(len(yRange)/(len(data[z + 3])) / 2)
                for array in x:
                    if not firstLayer: nextNodeNumber = 0
                    for value in array:
                        modelData.append((layer+nodeNumber, layer+layerSteps+nextNodeNumber, value))
                        try:
                            yPos = yRange[nodeNumber + yKor]
                        except IndexError:
                            yPos = yRange[nodeNumber]
                        if layer+nodeNumber not in pos:
                            pos[layer+nodeNumber] = np.array([xPos, yPos])
                        if layer+layerSteps+nextNodeNumber not in pos:
                            pos[layer+layerSteps+nextNodeNumber] = np.array([xPosNext, yRange[nextNodeNumber * yKorNext]])
                        if firstLayer:
                            nodeNumber += 1
                        else:
                            nextNodeNumber += 1
                    if not firstLayer:
                        nodeNumber += 1
                    else:
                        nextNodeNumber += 1

                layer += layerSteps
            z += 1
            firstLayer = False
        return modelData, pos

    def getPositionRanges(self, data):
        nOLayers = len(data)/2+1
        myMax = self.getLenOfFirstLayer(data[0])
        for x in data:
            if isinstance(x[0], np.float32):
                if len(x) > myMax:
                    myMax = len(x)
        xRange = np.arange(-1, 1.1, (2 / (nOLayers - 1)))
        yRange = np.arange(-1, 1.1, (2 / (myMax - 1)))
        return xRange, yRange

    def getLenOfFirstLayer(self, data):
        y = 0
        for x in data:
            y += len(x)
        return y
'''
        for x in data:
            nextNodeNumber = 0
            nodeNumber = 0
            if isinstance(x[0], np.ndarray):
                for array in x:
                    for value in array:
                        modelData.append((layer+nodeNumber, layer+layerSteps+ nextNodeNumber, value))
                        
                        nodeNumber += 1

                    nextNodeNumber += 1
                layer += layerSteps
'''