import os
import numpy as np
import pandas as pd

from imputation.data_treatment import BaseTreatment
from imputation.base_runner import BaseRunner


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/abalone.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def preprocess(self, values):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        values = values.copy()
        values[:, 0] = LabelEncoder().fit_transform(values[:, 0])
        values = OneHotEncoder(categorical_features=[0], sparse=False).fit_transform(values)
        return values
