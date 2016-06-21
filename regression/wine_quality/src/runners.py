import os
from sklearn.tree import DecisionTreeRegressor #noqa
from sklearn.ensemble import AdaBoostRegressor #noqa
from imputation.base_runner import (
    ImputationComparisonRunner, MissingComparisonRegressionRunner,
    MultipleNAttrStudy, MultipleEstimatorsStudy,
    MissingComparisonMulipleStudy, MissingComparisonMuliplePercentStudy,
    REGRESSION_ESTIMATORS
)


class WineImputationRunner(ImputationComparisonRunner):
    base_name = "Wine"
    estimator = DecisionTreeRegressor

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(WineImputationRunner, self).__init__([0, 4], source_data_file)

    @staticmethod
    def calc_score(estimator_instance, X, y):
        from sklearn.metrics import mean_absolute_error
        y_pred = estimator_instance.predict(X)
        return mean_absolute_error(y_pred, y)

    def edit_ax(self, ax):
        ax.legend(loc='upper left')
        ax.set_ylim(0, 1)
        ax.set_title("{} - {} - {}".format(self.base_name, self.estimator.__name__, self.n_attr_missing))


class WineMissingComparisonRunner(MissingComparisonRegressionRunner):
    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(WineMissingComparisonRunner, self).__init__(attrs_missing, [], source_data_file)


class WineMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = WineImputationRunner


class WineMultipleEstimators(MultipleEstimatorsStudy):
    runner = WineImputationRunner

    def get_iter(self):
        return REGRESSION_ESTIMATORS


class WineMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = WineMissingComparisonRunner


class WineMissingComparisonPercentStudy(MissingComparisonMuliplePercentStudy):
    runner = WineMissingComparisonRunner
