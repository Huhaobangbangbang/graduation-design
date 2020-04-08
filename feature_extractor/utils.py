from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def find_files(paths, extensions, sort=True):
    '''
    返回一个或多个目录中的文件列表
    '''
    if type(paths) is str:
        paths = [paths]
    files = []
    for path in paths:
        for file in os.listdir(path):
            if file.endswith(extensions):
                files.append(os.path.join(path, file))
    if sort:
        files.sort()
    return files

def fill_last_batch(excel_list, batch_size):
    '''
    用列表的最后一个示例填充最后一批。
    操作已执行到位。
    参数：图像列表：str列表，要填充的图像列表
    参数 批量: int, 批量大小

    '''
    num_examples = len(excel_list)
    num_batches = int(np.ceil(num_examples/batch_size))
    for i in range((num_batches*batch_size)-num_examples):
        excel_list.append(excel_list[-1])

def sort_feature_dataset(feature_dataset):
    '''
  当使用多个预处理线程时，特征数据集是不按文件名的字母顺序排序。这个函数对数据集进行就地排序，以便文件名和相应的fetaure按文件名排序。注意：分类已就位。
    :param feature_dataset: dict, containting filenames and all features
    :return:
    '''
    indices = np.argsort(feature_dataset['filenames'])
    feature_dataset['filenames'].sort()
    # Apply sorting to features for each image
    for key in feature_dataset.keys():
        if key == 'filenames': continue
        feature_dataset[key] = feature_dataset[key][indices]

def write_excel(filename, layer_names, feature_dataset):
    '''
    Writes features to excel file.
    参数：filename: str, filename to output
    参数 layer_names: list of str, layer names
    参数 feature_dataset: dict, containing features[layer_names] = vals

    '''
    with write_excel().File(filename, 'w') as hf:
        hf.create_dataset("filenames", data=feature_dataset['filenames'])
        for layer_name in layer_names:
            hf.create_dataset(layer_name, data=feature_dataset[layer_name], dtype=np.float32)


    plt.figure()

    plt.axis('off')
    plt.show()
