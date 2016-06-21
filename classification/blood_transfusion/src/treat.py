import os

from imputation.data_treatment import BaseTreatment


class CustomTreatment(BaseTreatment):

    def __init__(self, source_path=None, *args, **kwargs):
        dir_ = os.path.dirname(__file__)
        source_path = os.path.join(dir_, '../original/transfusion.data')
        super(CustomTreatment, self).__init__(source_path, *args, **kwargs)
