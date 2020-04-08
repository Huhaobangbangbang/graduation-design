

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
from datetime import datetime

from feature_extractor.feat_extract import FeatureExtractor
import feature_extractor.utils as utils

def feature_extraction_queue(feature_extractor, image_path, layer_names,
                             batch_size, num_classes, num_excels=100000):
    '''
   给定包含图像的目录，此函数提取特征所有图像。指定要从中提取特征的图层作为字符串列表。首先，我们寻找目录中的所有图像，
对列表进行排序并将其馈送到文件名队列。那么，批次是已处理和特征存储在对象“features”中。

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param layer_names: list of str, list of layer names
    :param batch_size: int, batch size
    :param num_classes: int, number of classes for ImageNet (1000 or 1001)
    :param num_images: int, number of images to process (default=100000)

    '''

    # 处理的文件位置信息
    excel_files = utils.find_files(image_path, ("xls", "csv"))
    num_excels = min(len(excel_files),num_excels)
    excel_files = excel_files[0:num_excels]

    num_examples = len(excel_files)
    num_batches = int(np.ceil(num_examples/batch_size))

    # Fill-up last batch so it is full (otherwise queue hangs)
    utils.fill_last_batch(excel_files, batch_size)#批量大小

    print("#"*80)
    print("Batch Size: {}".format(batch_size))
    print("Number of Examples: {}".format(num_examples))
    print("Number of Batches: {}".format(num_batches))

    # Add all the images to the filename queue
    feature_extractor.enqueue_excel_files(excel_files)

    # Initialize containers for storing processed filenames and features
    feature_dataset = {'filenames': []}
    for i, layer_name in enumerate(layer_names):
        layer_shape = feature_extractor.layer_size(layer_name)
        layer_shape[0] = len(excel_files)  # replace ? by number of examples
        feature_dataset[layer_name] = np.zeros(layer_shape, np.float32)
        print("Extracting features for layer '{}' with shape {}".format(layer_name, layer_shape))

    print("#"*80)

    # Perform feed-forward through the batches
    for batch_index in range(num_batches):

        t1 = time.time()

        # Feed-forward one batch through the network
        outputs = feature_extractor.feed_forward_batch(layer_names)

        for layer_name in layer_names:
            start = batch_index*batch_size
            end   = start+batch_size
            feature_dataset[layer_name][start:end] = outputs[layer_name]

        # Save the filenames of the images in the batch
        feature_dataset['filenames'].extend(outputs['filenames'])

        t2 = time.time()
        examples_in_queue = outputs['examples_in_queue']
        examples_per_second = batch_size/float(t2-t1)

        print("[{}] Batch {:04d}/{:04d}, Batch Size = {}, Examples in Queue = {}, Examples/Sec = {:.2f}".format(
            datetime.now().strftime("%Y-%m-%d %H:%M"), batch_index+1,
            num_batches, batch_size, examples_in_queue, examples_per_second
        ))

    # If the number of pre-processing threads >1 then the output order is
    # non-deterministic. Therefore, we order the outputs again by filenames so
    # the images and corresponding features are sorted in alphabetical order.
    if feature_extractor.num_preproc_threads > 1:
        utils.sort_feature_dataset(feature_dataset)

    #因为已经装满了我们把最后一部分切断了
    feature_dataset['filenames'] = feature_dataset['filenames'][0:num_examples]
    for layer_name in layer_names:
        feature_dataset[layer_name] = feature_dataset[layer_name][0:num_examples]

    return feature_dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TensorFlow feature extraction")

    parser.add_argument("--out_file", dest="out_file", type=str, default="./features.h5", help="path to save features (HDF5 file)")

    parser.add_argument("--preproc_func", dest="preproc_func", type=str, default=None, help="force the image preprocessing function (None)")

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size (32)")
    parser.add_argument("--num_classes", dest="num_classes", type=int, default=1001, help="number of classes (1001)")
    args = parser.parse_args()

    # resnet_v2_101/logits,resnet_v2_101/pool4 => to list of layer names


    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(

        num_classes=args.num_classes,
        preproc_func_name=args.preproc_func,
        preproc_threads=args.num_preproc_threads
    )

    # Print the network summary, use these layer names for feature extraction


    # Feature extraction example using a filename queue to feed images
    feature_dataset = feature_extraction_queue(
        feature_extractor, args.image_path,
        args.batch_size, args.num_classes)

    # 将属性写为Excel file
    utils.h5py(args.out_file, feature_dataset)
    print("Successfully written features to: {}".format(args.out_file))

    # Close the threads and close session.
    feature_extractor.close()
    print("Finished.")
