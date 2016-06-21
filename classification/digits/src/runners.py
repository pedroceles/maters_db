import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy
from treat import CATEGORICAL_DATA


class DigitImputationRunner(ImputationComparisonRunner):
    base_name = "Digit"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(DigitImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    def edit_ax(self, ax):
        super(DigitImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class DigitMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(DigitMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class DigitMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = DigitImputationRunner


class DigitMultipleEstimators(MultipleEstimatorsStudy):
    runner = DigitImputationRunner


class DigitMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = DigitMissingComparisonRunner
