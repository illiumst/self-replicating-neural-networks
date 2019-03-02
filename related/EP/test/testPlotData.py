import unittest
import numpy as np

from src.PltData import PltData

class TestPlotData(unittest.TestCase):

    def testPlotNNModel(self):
        #[2, 3, 5] Netz
        nn = np.array([[-0.00862074, -0.00609563], [ 0.03935056,  0.0159397 ]], dtype=np.float32),\
                      np.array([ 0.,  0.], dtype=np.float32),\
                      np.array([[ 0.01351449,  0.04824072,  0.04954299], [ 0.04268739, -0.04188565,  0.03875775]], dtype=np.float32),\
                      np.array([ 0.,  0.,  0.], dtype=np.float32),\
                      np.array([[ 0.01074128, -0.00355459,  0.00787288, -0.02870593, -0.0204265 ], [ 0.01399798, -0.0096233 ,  0.03152497,  0.03874204, -0.0466414 ], [ 0.04445429, -0.02976017,  0.00065653, -0.04210887, -0.02864893]], dtype=np.float32),\
                      np.array([ 0.,  0.,  0.,  0.,  0.], dtype=np.float32)

        #[2, 1, 2] Netz
        nn2 =np.array([[ 0.01390548, -0.01149112], [ 0.02786468, -0.02605006]], dtype=np.float32), \
             np.array([ 0.,  0.], dtype=np.float32), \
             np.array([[-0.03265964],[ 0.013609  ]], dtype=np.float32), \
             np.array([ 0.], dtype=np.float32), \
             np.array([[ 0.02287653,  0.02650055]], dtype=np.float32), \
             np.array([ 0.,  0.], dtype=np.float32)

        #[4,2,2]
        nn3 = np.array([[ 0.03519103, -0.04059422,  0.04508766, -0.04067679], [ 0.01457861,  0.01178179, -0.01784203,  0.00051603], [-0.00807861,  0.01152407,  0.0136507 ,  0.02639047], [ 0.04526602, -0.01604335,  0.00661949,  0.0434478 ]], dtype=np.float32), \
                       np.array([ 0.,  0.,  0.,  0.], dtype=np.float32),\
                       np.array([[ 0.03728329, -0.01507163], [ 0.00789828,  0.0494065 ], [-0.00945786, -0.04301547], [-0.01999701, -0.01306728]], dtype=np.float32),\
                       np.array([ 0.,  0.], dtype=np.float32),\
                       np.array([[-0.03051615, -0.03279487],  [ 0.01100482, -0.02652025]], dtype=np.float32),\
                       np.array([ 0.,  0.], dtype=np.float32)

        # [1, 1, 2] Netz
        nn4 = np.array([[0.01390548]], dtype=np.float32), \
              np.array([0.], dtype=np.float32), \
              np.array([[-0.03265964]], dtype=np.float32), \
              np.array([0.], dtype=np.float32), \
              np.array([[0.02287653, 0.02650055]], dtype=np.float32), \
              np.array([0., 0.], dtype=np.float32)

        PltData(None).plotNNModel(nn3, "test.png")