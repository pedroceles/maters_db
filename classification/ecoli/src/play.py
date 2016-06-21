# -*- coding: utf-8 -*-
import os

from imputation.data_treatment import BaseTreatment


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/ecoli.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def preprocess(self, values):
        from sklearn.preprocessing import LabelEncoder
        values = values.copy()[:, 1:]
        values[:, -1] = LabelEncoder().fit_transform(values[:, -1])
        return values
