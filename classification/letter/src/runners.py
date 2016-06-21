import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy
from treat import CATEGORICAL_DATA


class LetterImputationRunner(ImputationComparisonRunner):
    base_name = "Letter"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(LetterImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    def edit_ax(self, ax):
        super(LetterImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class LetterMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(LetterMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class LetterMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = LetterImputationRunner


class LetterMultipleEstimators(MultipleEstimatorsStudy):
    runner = LetterImputationRunner


class LetterMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = LetterMissingComparisonRunner
