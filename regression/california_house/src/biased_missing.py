# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from imputation.bias import BiasedMissing, p_factory_gaussian
from imputation.base_runner import BaseRunner

from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor


def run(attrs_missing=[8, 9]):
    from sklearn.cross_validation import train_test_split
    imp = Imputer()
    r = BaseRunner('../treated.csv')
    values = np.copy(r.values)
    X, y = values[:, :-1], values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    est = DecisionTreeRegressor()

    ps = []
    for attr_missing in attrs_missing:
        p = p_factory_gaussian([X_train[:, attr_missing].std()], [0.2])
        ps.append(p)
    bm = BiasedMissing(X_train, attrs_missing, ps)
    bm.hide_data()
    backbones = set(range(X_train.shape[1]))
    backbones = list(backbones.difference(attrs_missing))
    filled_data = bm.input_data_neighbors(5, backbones)

    imputed_vales = bm.hided_data
    imputed_vales = imp.fit_transform(imputed_vales)

    fig, axes = plt.subplots(1, len(attrs_missing) + 1)

    axes_hist = axes[:-1]
    ax_bar = axes[-1]
    original_color = 'blue'
    neigh_color = 'green'
    mean_color = 'red'
    for attr_missing, ax_hist in zip(attrs_missing, axes_hist):
        original_data = X_train[:, attr_missing]
        range_ = (original_data.min(), original_data.max())
        ax_hist.hist(X_train[:, attr_missing], range=range_, label="Original", color=original_color)
        ax_hist.hist(filled_data[:, attr_missing], range=range_, alpha=0.5, label="Imputado Vizinhos", color=neigh_color)
        ax_hist.hist(imputed_vales[:, attr_missing], range=range_, alpha=0.5, label=u"Imputado Média", color=mean_color)
        ax_hist.set_yticklabels([])
        ax_hist.set_title(attr_missing)
        ax_hist.legend()

    # error_complete = run_estimator(est, X_train[:, attrs_missing], y_train, X_test[:, attrs_missing], y_test)
    # error_neigh = run_estimator(est, filled_data[:, attrs_missing], y_train, X_test[:, attrs_missing], y_test)
    # error_mean = run_estimator(est, imputed_vales[:, attrs_missing], y_train, X_test[:, attrs_missing], y_test)

    error_complete = run_estimator(est, X_train, y_train, X_test, y_test)
    error_neigh = run_estimator(est, filled_data, y_train, X_test, y_test)
    error_mean = run_estimator(est, imputed_vales, y_train, X_test, y_test)

    ax_bar.bar([1, 2, 3], [error_complete, error_neigh, error_mean], color=[original_color, neigh_color, mean_color])
    ax_bar.set_xticks([1, 2, 3])
    ax_bar.set_xticklabels(["Original", "Imputado Vizinhos", u"Imputado Média"])
    ax_bar.set_ylabel("MAE")
    ax_bar.set_title(est.__class__.__name__)
    plt.show()


def train_estimator(est, X_train, y_train):
    est.fit(X_train, y_train)


def calc_score(est, X_test, y_test):
    from sklearn.metrics import mean_absolute_error
    y_pred = est.predict(X_test)
    return mean_absolute_error(y_test, y_pred)


def run_estimator(est, X_train, y_train, X_test, y_test):
    train_estimator(est, X_train, y_train)
    return calc_score(est, X_test, y_test)

if __name__ == '__main__':
    run()
