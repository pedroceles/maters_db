import os
from imputation.base_runner import (
    ImputationComparisonRunner, MultipleNAttrStudy, MultipleEstimatorsStudy,
    MissingComparisonClassificationRunner, MissingComparisonMulipleStudy
)


class BloodImputationRunner(ImputationComparisonRunner):
    base_name = "Blood Transf."

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(BloodImputationRunner, self).__init__([], source_data_file)


class BloodMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = BloodImputationRunner


class BloodMultipleEstimators(MultipleEstimatorsStudy):
    runner = BloodImputationRunner


class BloodMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(BloodMissingComparisonRunner, self).__init__(attrs_missing, [], source_data_file)


class BloodMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = BloodMissingComparisonRunner
