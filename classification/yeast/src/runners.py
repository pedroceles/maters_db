import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy
from treat import CATEGORICAL_DATA
CATEGORICAL_DATA = CATEGORICAL_DATA[:-1]


class YeastImputationRunner(ImputationComparisonRunner):
    base_name = "Yeast"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(YeastImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    def edit_ax(self, ax):
        super(YeastImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class YeastMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(YeastMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class YeastMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = YeastImputationRunner


class YeastMultipleEstimators(MultipleEstimatorsStudy):
    runner = YeastImputationRunner


class YeastMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = YeastMissingComparisonRunner
