import os
from sklearn.tree import DecisionTreeRegressor #noqa
from sklearn.ensemble import AdaBoostRegressor #noqa
from sklearn.linear_model import SGDRegressor #noqa
from imputation.base_runner import (
    ImputationComparisonRunner, MissingComparisonRegressionRunner,
    MultipleNAttrStudy, MultipleEstimatorsStudy,
    MissingComparisonMulipleStudy, MissingComparisonMuliplePercentStudy,
    REGRESSION_ESTIMATORS
)


class WankaraImputationRunner(ImputationComparisonRunner):
    base_name = "Wankara"
    estimator = DecisionTreeRegressor
    test_split = 0.1

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(WankaraImputationRunner, self).__init__([], source_data_file)

    @staticmethod
    def calc_score(estimator_instance, X, y):
        from sklearn.metrics import mean_absolute_error
        y_pred = estimator_instance.predict(X)
        return mean_absolute_error(y_pred, y)

    def edit_ax(self, ax):
        ax.legend(loc='upper left')
        ax.set_ylim(0, 200)
        ax.set_title("{} - {} - {}".format(self.base_name, self.estimator.__name__, self.n_attr_missing))


class WankaraMissingComparisonRunner(MissingComparisonRegressionRunner):
    estimator = DecisionTreeRegressor

    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(WankaraMissingComparisonRunner, self).__init__(attrs_missing, [], source_data_file)


class WankaraMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = WankaraImputationRunner


class WankaraMultipleEstimators(MultipleEstimatorsStudy):
    runner = WankaraImputationRunner

    def get_iter(self):
        return REGRESSION_ESTIMATORS


class WankaraMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = WankaraMissingComparisonRunner


class WankaraMissingComparisonPercentStudy(MissingComparisonMuliplePercentStudy):
    runner = WankaraMissingComparisonRunner
