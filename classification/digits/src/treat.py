import os

from imputation.data_treatment import BaseTreatment

CATEGORICAL_DATA = range(256)


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/semeion.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)

    def read_file(self, *args, **kwargs):
        df = super(CustomTreatment, self).read_file(sep=' ')
