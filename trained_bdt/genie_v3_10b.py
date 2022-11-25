import cppyy
import numpy as np
import os


class GENIEv3_10b_BDT:
    def __init__(self):
        filename = 'GeneratorReweight_GENIEv3_G18_10b_00_000.h'
        parent_dir = 'trained_bdt/'
        cppyy.include(os.path.join(parent_dir, filename))
        self.reweighter = cppyy.gbl.GeneratorReweight_GENIEv3_G18_10b_00_000()
    def predict(self, array):
        return np.array([self.reweighter.predict(row, 1) for row in array])
