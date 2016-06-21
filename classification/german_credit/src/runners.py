import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy
from treat import CATEGORICAL_DATA


class GCreditImputationRunner(ImputationComparisonRunner):
    base_name = "GCredit"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(GCreditImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    def edit_ax(self, ax):
        super(GCreditImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class GCreditMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(GCreditMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class GCreditMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = GCreditImputationRunner


class GCreditMultipleEstimators(MultipleEstimatorsStudy):
    runner = GCreditImputationRunner


class GCreditMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = GCreditMissingComparisonRunner
