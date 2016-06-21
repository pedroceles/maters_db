import os

from imputation.data_treatment import BaseTreatment

CATEGORICAL_DATA = [1, 3, 5, 6, 7, 8, 9, 13, -1]



class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/adult.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def preprocess(self, values):
        from sklearn.preprocessing import LabelEncoder
        values = values.copy()
        # Reitrando missing data
        missing_index = (values == '?').any(axis=1)
        values = values[~missing_index]
        for attr in CATEGORICAL_DATA:
            values[:, attr] = LabelEncoder().fit_transform(values[:, attr])
        return values

    def read_file(self, *args, **kwargs):
        return super(CustomTreatment, self).read_file(sep=r',\s+')
