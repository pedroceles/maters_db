import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonRegressionRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy


class AbaloneImputationRunner(ImputationComparisonRunner):
    base_name = "Abalone"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(AbaloneImputationRunner, self).__init__([0, 1, 2], source_data_file)

    def edit_ax(self, ax):
        super(AbaloneImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class AbaloneMissingComparisonRunner(MissingComparisonRegressionRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(AbaloneMissingComparisonRunner, self).__init__(attrs_missing, [0, 1, 2], source_data_file)


class AbaloneMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = AbaloneImputationRunner


class AbaloneMultipleEstimators(MultipleEstimatorsStudy):
    runner = AbaloneImputationRunner


class AbaloneMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = AbaloneMissingComparisonRunner
