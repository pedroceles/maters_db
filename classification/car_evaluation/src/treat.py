import os
import numpy as np
import pandas as pd

from imputation.data_treatment import BaseTreatment
from imputation.base_runner import BaseRunner


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/car.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def preprocess(self, values):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        values = values.copy()
        for attr in range(values.shape[1]):
            values[:, attr] = LabelEncoder().fit_transform(values[:, attr])
        return values
