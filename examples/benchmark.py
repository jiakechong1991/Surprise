# -*- coding=utf-8 -*-

"""
This module runs a 5-Fold CV for all the algorithms (default parameters) on
the movielens datasets, and reports average RMSE, MAE, and total computation
time.  It is used for making tables in the README.md file
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import datetime
import random
import os
import sys

UP_FOLDER = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(UP_FOLDER)
import numpy as np
import six
from tabulate import tabulate

from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering

# The algorithms to cross-validate
classes = [
    # BaselineOnly,  # 1
    SVD,  # 2
    # SVDpp,  # 3
    # NMF,  # 4
    # SlopeOne,  # 5
    # KNNBasic,  # 6
    # KNNWithMeans,  # 7
    # KNNBaseline,  # 8
    # CoClustering,  # 9
    # NormalPredictor  # 10
]

# ugly dict to map algo names and datasets to their markdown links in the table
stable = 'http://surprise.readthedocs.io/en/stable/'
LINK = {'SVD': '[{}]({})'.format('SVD',
                                 stable +
                                 'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD'),
        'SVDpp': '[{}]({})'.format('SVD++',
                                   stable +
                                   'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp'),
        'NMF': '[{}]({})'.format('NMF',
                                 stable +
                                 'matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF'),
        'SlopeOne': '[{}]({})'.format('Slope One',
                                      stable +
                                      'slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne'),
        'KNNBasic': '[{}]({})'.format('k-NN',
                                      stable +
                                      'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic'),
        'KNNWithMeans': '[{}]({})'.format('Centered k-NN',
                                          stable +
                                          'knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans'),
        'KNNBaseline': '[{}]({})'.format('k-NN Baseline',
                                         stable +
                                         'knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline'),
        'CoClustering': '[{}]({})'.format('Co-Clustering',
                                          stable +
                                          'co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering'),
        'BaselineOnly': '[{}]({})'.format('Baseline',
                                          stable +
                                          'basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly'),
        'NormalPredictor': '[{}]({})'.format('Random',
                                             stable +
                                             'basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor'),
        'ml-100k': '[{a}]'.format(a='Movielens 100k'),  # http://grouplens.org/datasets/movielens/100k
        'ml-1m': '[{a}]'.format(a='Movielens 1M'),  # http://grouplens.org/datasets/movielens/1m
        }


# set RNG
np.random.seed(0)
random.seed(0)

dataset = 'ml-1m'
# dataset = 'ml-100k'
data = Dataset.load_builtin(dataset)
print(dir(data))
print(data)
print(len(data.raw_ratings))
print(data.ratings_file)
print(data.reader.indexes)
# 1/0
kf = KFold(n_splits=5, random_state=0)  # folds will be the same for all algorithms.

table = []
for klass in classes:
    start = time.time()
    out = cross_validate(klass(), data, ['rmse', 'mae'], kf, verbose=True)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))

    new_line = [klass.__name__, mean_rmse, mean_mae, cv_time]
    table.append(new_line)
    print(LINK[klass.__name__])  # 打印算法链接


header = [dataset, 'RMSE', 'MAE', 'Time']
print(tabulate(table, header, tablefmt="pipe"))
