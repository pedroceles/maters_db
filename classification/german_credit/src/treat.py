import os

from imputation.data_treatment import BaseTreatment

NUMERICAL_DATA = [1, 4, 7, 10, 12, 15, 17]
CATEGORICAL_DATA = set(range(20)).difference(NUMERICAL_DATA)
CATEGORICAL_DATA = sorted(CATEGORICAL_DATA)


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/german.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def preprocess(self, values):
        from sklearn.preprocessing import LabelEncoder
        values = values.copy()
        for attr in CATEGORICAL_DATA:
            values[:, attr] = LabelEncoder().fit_transform(values[:, attr])
        return values

    def read_file(self, *args, **kwargs):
        return super(CustomTreatment, self).read_file(sep=' ')
