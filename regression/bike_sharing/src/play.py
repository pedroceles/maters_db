import os

import pandas as pd

from imputation.data_treatment import BaseTreatment


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/hour.csv')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def read_file(self):
        self._df = pd.read_csv(self._source_path, index_col=0, usecols=[0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16])
        return self._df
