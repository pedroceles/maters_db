# -*- coding: utf-8 -*-
import numpy as np
from runners import BikeMissingComparisonRunner
from imputation.estimators import (
    ReducedModelRegressor, OriginalReducedModelRegressor
)
from imputation.base_runner import mape
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


def run():
    r = BikeMissingComparisonRunner([3, 2, 1])
    est1 = ReducedModelRegressor(DecisionTreeRegressor)
    est2 = OriginalReducedModelRegressor(DecisionTreeRegressor)
    X, y = r.values[:, :-1], r.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    est1.fit(X_train, y_train)
    est2.fit(X_train, y_train)
    indexes = np.arange(X_test.shape[0])
    indexes_to_none = np.random.choice(indexes, int(indexes[-1] * 0.5), False)
    X_test[indexes_to_none, 3] = None
    indexes_to_none = np.random.choice(indexes, int(indexes[-1] * 0.5), False)
    X_test[indexes_to_none, 2] = None
    y_pred1 = est1.predict(X_test)
    y_pred2 = est2.predict(X_test)
    return mape(y_test, y_pred1), mape(y_test, y_pred2)
