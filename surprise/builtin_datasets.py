# -*- coding=utf-8 -*-

"""This module contains built-in datasets that can be automatically
downloaded."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves.urllib.request import urlretrieve
import zipfile
from collections import namedtuple
import os
from os.path import join


# 获取或者创建存放数据集的目录
def get_dataset_dir():
    '''Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.surprise_data/, but it can also be set by the
    environment variable ``SURPRISE_DATA_FOLDER``.
    '''

    folder = os.environ.get('SURPRISE_DATA_FOLDER', os.path.expanduser('~') +
                            '/.surprise_data/')
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - the parameters of the corresponding reader
# 创建一个便捷的类
BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'reader_params'])


# 这是内建支持的数据集
BUILTIN_DATASETS = {
    'ml-100k':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',  # 要下载的数据集
            path=join(get_dataset_dir(), 'ml-100k/ml-100k/u.data'),  # 用户-商品-评分-文件
            reader_params=dict(line_format='user item rating timestamp',  # 这个文件的行数据格式
                               rating_scale=(1, 5),  # 评分的等级范围：1-5分
                               sep='\t')  # 行数据的切割符
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=join(get_dataset_dir(), 'ml-1m/ml-1m/ratings.dat'),
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
    'jester':
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip',
            path=join(get_dataset_dir(), 'jester/jester_ratings.dat'),
            reader_params=dict(line_format='user item rating',
                               rating_scale=(-10, 10))
        )
}


# 下载指定名称的数据集
def download_builtin_dataset(name):

    dataset = BUILTIN_DATASETS[name]

    print('Trying to download dataset from ' + dataset.url + '...')
    tmp_file_path = join(get_dataset_dir(), 'tmp.zip')
    # 把指定的URL文件下载成本地的指定文件
    urlretrieve(dataset.url, tmp_file_path)

    # 解压所文件
    with zipfile.ZipFile(tmp_file_path, 'r') as tmp_zip:
        tmp_zip.extractall(join(get_dataset_dir(), name))

    os.remove(tmp_file_path)
    print('Done! Dataset', name, 'has been saved to',
          join(get_dataset_dir(), name))
