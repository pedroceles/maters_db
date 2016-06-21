import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy
from treat import CATEGORICAL_DATA
CATEGORICAL_DATA = CATEGORICAL_DATA[:-1]


class AdultImputationRunner(ImputationComparisonRunner):
    base_name = "Adult"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(AdultImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    def edit_ax(self, ax):
        super(AdultImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class AdultMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(AdultMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class AdultMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = AdultImputationRunner


class AdultMultipleEstimators(MultipleEstimatorsStudy):
    runner = AdultImputationRunner


class AdultMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = AdultMissingComparisonRunner
