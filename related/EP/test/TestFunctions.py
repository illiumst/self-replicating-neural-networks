import unittest
import numpy as np

import src.Functions

class TestFunctions(unittest.TestCase):

    def testcalcMeanSquaredError(self):
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1.1, 2.05, 2.95, 4.01, 4.5])
        self.assertEqual(0.05, src.Functions.calcMeanSquaredError(a, b))

        a = np.array(['1', '2', '3', '4', '5'])
        b = np.array(['1.1', '2.05', '2.95', '4.01', '4.5'])
        self.assertEqual(0.05, src.Functions.calcMeanSquaredError(a, b))

    def testGetRandomLayer(self):
        layer = (1, 3)
        self.assertEqual(layer, np.shape(src.Functions.getRandomLayer(layer)))
        layer = (3, 1)
        self.assertEqual(layer, np.shape(src.Functions.getRandomLayer(layer)))
        layer = (8, 2)
        self.assertEqual(layer, np.shape(src.Functions.getRandomLayer(layer)))
        layer = (100, 1)
        self.assertEqual(layer, np.shape(src.Functions.getRandomLayer(layer)))
        layer = (1, 1)
        self.assertEqual(layer, np.shape(src.Functions.getRandomLayer(layer)))
        layer = (4, 50)
        self.assertEqual(layer, np.shape(src.Functions.getRandomLayer(layer)))
