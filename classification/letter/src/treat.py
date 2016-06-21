import os

from imputation.data_treatment import BaseTreatment

CATEGORICAL_DATA = []


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/letter-recognition.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def preprocess(self, values):
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        values = values.copy()
        y, X = values[:, 0], values[:, 1:]
        y = LabelEncoder().fit_transform(y)
        return np.c_[X, y]
