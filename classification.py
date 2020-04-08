

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from scipy import misc

from feature_extractor.feat_extract import   FeatureExtractor
import feature_extractor.utils as utils


def classification_queue_input(feature_extractor, image_path, logits_name,
                               batch_size, num_classes):
    '''
    使用预先训练的图像分类的示例函数网络。这将测试文件名队列作为输入方法。列出了要处理的图像文件，这些文件将被输入文件名队列，然后图像从队列中出列并分类。

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param logits_name: str, name of logits layer in network
    :param batch_size: int, batch size
    :param num_classes: int, number of classes for ImageNet (1000 or 1001)
    :return:
    '''

    # Add a list of images to process
    excel_files = utils.find_files(image_path, ("csv", "xls"))

    # Push the images through the network
    feature_extractor.enqueue_image_files(excel_files)
    outputs = feature_extractor.feed_forward_batch([logits_name], fetch_images=True)

    # Compute the predictions, note that we asume layer_names[0] corresponds to logits
    predictions = np.squeeze(outputs[logits_name])
    predictions = np.argmax(predictions, axis=1)

    for i in range(batch_size):
        image = misc.imread(excel_files[i])
        class_index = predictions[i] if num_classes == 1001 else predictions[i]+1
        utils.display_imagenet_prediction(image, class_index)


def classification_placeholder_input(feature_extractor, data, logits_name,
                                     batch_size, num_classes):
    '''
使用预先训练的图像分类的示例函数网络。这个函数使用占位符测试简单的输入方法。
它从磁盘加载一批图像，使用Inception对它们进行预处理预处理，然后通过网络转发。输入图像完成后，将显示预测的ImageNet类。

    :param feature_extractor: object, TF feature extractor
    :param image_path: str, path to directory containing images
    :param logits_name: str, name of logits layer in network
    :param batch_size: int, batch size
    :param num_classes: int, number of classes for ImageNet (1000 or 1001)
    :return:
    '''

    # Add a list of images to process
    excel_files = utils.find_files(data, ("csv", "xls"))

    # Load one batch of images
    batch_images = np.zeros([batch_size, feature_extractor.image_size,
                             feature_extractor.image_size, 3], dtype=np.float32)

    for i in range(batch_size):
        # Note: this corresponds to 'inception' preprocessing. You don't need
        # this when using the queues as input pipeline, since the get_preprocessing()
        # function automatically determines it.
        image = misc.imread(excel_files[i])
        image = misc.imresize(
            image, (feature_extractor.image_size, feature_extractor.image_size))
        image = (image/255.0).astype(dtype=np.float32)
        image -= 0.5
        image *= 2.0
        batch_images[i] = image

    # Push the images through the network
    outputs = feature_extractor.feed_forward_batch(
        [logits_name], batch_images, fetch_images=True)

    # Compute the predictions, note that we asume layer_names[0] corresponds to logits
    predictions = np.squeeze(outputs[logits_name])
    predictions = np.argmax(predictions, axis=1)

    # Display predictions
    for i in range(batch_size):
        image = (((batch_images[i]/2.0)+0.5)*255.0).astype(np.uint8)
        class_index = predictions[i] if num_classes == 1001 else predictions[i]+1
        utils.display_imagenet_prediction(image, class_index)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TensorFlow feature extraction")
    parser.add_argument("--network", dest="network_name", type=str, required=True, help="model name, e.g. 'resnet_v2_101'")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, required=True, help="path to pre-trained checkpoint file")
    parser.add_argument("--image_path", dest="image_path", type=str, required=True, help="path to directory containing images")
    parser.add_argument("--logits_name", dest="logits_name", type=str, required=True, help="name of logits layer in network")
    parser.add_argument("--preproc_func", dest="preproc_func", type=str, default=None, help="force the image preprocessing function (None)")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="batch size (32)")
    parser.add_argument("--num_classes", dest="num_classes", type=int, default=1001, help="number of classes (1001)")
    args = parser.parse_args()

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(
        network_name=args.network_name,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        preproc_func_name=args.preproc_func)

    # Print the network summary, use these layer names for feature extraction
    feature_extractor.print_network_summary()

    # OPTION 1. Test image classification using a filename queue to feed images
    classification_queue_input(
        feature_extractor, args.image_path, args.logits_name,
        args.batch_size, args.num_classes)

    # OPTION 2. Test image classification by manually feeding images into placeholders
    classification_placeholder_input(
        feature_extractor, args.image_path, args.logits_name,
        args.batch_size, args.num_classes)
