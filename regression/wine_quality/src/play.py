import os

import pandas as pd

from imputation.data_treatment import BaseTreatment


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/winequality-red.csv')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def read_file(self):
        self._df = pd.read_csv(self._source_path, sep=';')
        return self._df
