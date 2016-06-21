import os
from imputation.base_runner import ImputationComparisonRunner, MissingComparisonClassificationRunner, MultipleNAttrStudy, MultipleEstimatorsStudy, MissingComparisonMulipleStudy


class CarImputationRunner(ImputationComparisonRunner):
    base_name = "Car"
    test_split = 0.8

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(CarImputationRunner, self).__init__(range(6), source_data_file)

    def edit_ax(self, ax):
        super(CarImputationRunner, self).edit_ax(ax)
        ax.legend(loc='upper right')


class CarMissingComparisonRunner(MissingComparisonClassificationRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(CarMissingComparisonRunner, self).__init__(attrs_missing, range(6), source_data_file)


class CarMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = CarImputationRunner


class CarMultipleEstimators(MultipleEstimatorsStudy):
    runner = CarImputationRunner


class CarMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = CarMissingComparisonRunner
