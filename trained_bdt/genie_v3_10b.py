import cppyy
import numpy as np
cppyy.include('/data/rradev/generator_translation/trained_bdt/GeneratorReweight_GENIEv3_G18_10b_00_000.h')


class GENIEv3_10b_BDT:
    def __init__(self):
        self.reweighter = cppyy.gbl.GeneratorReweight_GENIEv3_G18_10b_00_000()
    def predict(self, array):
        return np.array([self.reweighter.predict(row, 1) for row in array])
