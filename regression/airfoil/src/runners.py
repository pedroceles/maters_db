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

from treat import CATEGORICAL_DATA


class AirfoilImputationRunner(ImputationComparisonRunner):
    base_name = "Airfoil"
    estimator = DecisionTreeRegressor
    test_split = 0.1

    def __init__(self):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(AirfoilImputationRunner, self).__init__(CATEGORICAL_DATA, source_data_file)

    @staticmethod
    def calc_score(estimator_instance, X, y):
        from sklearn.metrics import mean_absolute_error
        y_pred = estimator_instance.predict(X)
        return mean_absolute_error(y_pred, y)

    def edit_ax(self, ax):
        ax.legend(loc='upper left')
        ax.set_ylim(0, 200)
        ax.set_title("{} - {} - {}".format(self.base_name, self.estimator.__name__, self.n_attr_missing))


class AirfoilMissingComparisonRunner(MissingComparisonRegressionRunner):
    estimator = DecisionTreeRegressor

    def __init__(self, attrs_missing):
        source_data_file = os.path.join(os.path.dirname(__file__), '../treated.csv')
        super(AirfoilMissingComparisonRunner, self).__init__(attrs_missing, CATEGORICAL_DATA, source_data_file)


class AirfoilMultipleNAttrImputationStudy(MultipleNAttrStudy):
    runner = AirfoilImputationRunner


class AirfoilMultipleEstimators(MultipleEstimatorsStudy):
    runner = AirfoilImputationRunner

    def get_iter(self):
        return REGRESSION_ESTIMATORS


class AirfoilMissingComparisonStudy(MissingComparisonMulipleStudy):
    runner = AirfoilMissingComparisonRunner


class AirfoilMissingComparisonPercentStudy(MissingComparisonMuliplePercentStudy):
    runner = AirfoilMissingComparisonRunner
