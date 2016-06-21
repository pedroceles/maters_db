import os
from sklearn import tree
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy


class EcoliImputationRunner(ImputationComparisonRunner):
    base_name = "Ecoli"

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(EcoliImputationRunner, self).__init__([3, 4], source_data_file)


class EcoliMissingComparisonRunner(MissingComparisonClassificationRunner):
    classification = True
    estimator = tree.DecisionTreeClassifier

    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(EcoliMissingComparisonRunner, self).__init__(attrs_missing, [3, 4], source_data_file)


class EcoliMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = EcoliImputationRunner


class EcoliMultipleEstimators(MultipleEstimatorsStudy):
    runner = EcoliImputationRunner


class EcoliMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = EcoliMissingComparisonRunner
