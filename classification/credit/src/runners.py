import os
from imputation.base_runner import (
    ImputationComparisonRunner, MultipleNAttrStudy, MultipleEstimatorsStudy,
    MissingComparisonClassificationRunner, MissingComparisonMulipleStudy
)


class CreditImputationRunner(ImputationComparisonRunner):
    base_name = "Credit"

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(CreditImputationRunner, self).__init__([0, 3, 4, 5, 7, 8, 10, 11], source_data_file)


class CreditMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = CreditImputationRunner


class CreditMultipleEstimators(MultipleEstimatorsStudy):
    runner = CreditImputationRunner


class CreditMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(CreditMissingComparisonRunner, self).__init__(attrs_missing, [], source_data_file)


class CreditMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = CreditMissingComparisonRunner
