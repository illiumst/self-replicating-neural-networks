import unittest
import numpy as np

from src.FeatureReduction import FeatureReduction

class TestFeatureReduction(unittest.TestCase):

    def testfft(self):
        data = np.array([1,2,3,4])
        d = FeatureReduction("mean").mean(FeatureReduction('mean').shuffelVec(data,4),2)
        print(d)

    def testVecMean(self):
        data = np.array([1,2,3,4,5,6,7,8,9])
        d = FeatureReduction("mean").mean(data, 1)
        self.assertEqual(np.array([45/9]),d)

        d = FeatureReduction("mean").mean(data, 2)
        np.testing.assert_array_equal(np.array([round(12.5/4.5,6), round(32.5/4.5,6)]), d)

        d = FeatureReduction("mean").mean(data, 3)
        np.testing.assert_array_equal(np.array([2, 5, 8]), d)

        d = FeatureReduction("mean").mean(data, 4)
        np.testing.assert_array_equal(np.array([round(3.75/2.25,6), round(8.75/2.25,6), round(13.75 / 2.25,6), round(18.75/2.25,6)]), d)

        d = FeatureReduction("mean").mean(data, 5)
        np.testing.assert_array_equal(np.array([round(2.6 / 1.8,6), round(5.8 / 1.8,6), round(9 / 1.8,6),
                                                round(12.2 / 1.8,6), round(15.4/1.8,6)]), d)

        d = FeatureReduction("mean").mean(data, 6)
        np.testing.assert_array_equal(np.array([round(2 / 1.5,6), round(4 / 1.5,6), round(6.5 / 1.5,6),
                                                round(8.5 / 1.5,6), round(11/1.5,6),round(13/1.5,6)]), d)

        d = FeatureReduction("mean").mean(data, 9)
        np.testing.assert_array_equal(np.array([1,2,3,4,5,6,7,8,9]), d)

    def testWeigthsToVec(self):
        test =np.array([[ 0.04457645, -0.03319572]], dtype=np.float32), np.array([ 0.,  0.], dtype=np.float32), np.array([[-0.03747094],
       [ 0.01189486]], dtype=np.float32), np.array([ 0.], dtype=np.float32)
        FeatureReduction("mean").calc(test, 1)

    def testShuffelVec(self):
        vec = np.array([1,2,3,4,5,6,7,8,9,10])
        print(FeatureReduction('mean').shuffelVec(vec,2))

    def testPP(self):
        vec = np.array([1., 5., 3.])
        print(FeatureReduction('mean').calc(vec, 1))