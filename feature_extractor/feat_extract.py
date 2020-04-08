from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#加入绝对引入特性，引入future语句 启用相对导入等特性
import tensorflow as tf

#使用预训练的TF-slim模型
slim = tf.contrib.slim
#训练一个featurextractor模型
class FeatureExtractor(object):

    def __init__(self, network_name, checkpoint_path, batch_size, num_classes,
                 preproc_func_name=None, preproc_threads=2):
#使用tf.slim和models/slim的TensorFlow特征提取器。核心功能是加载网络架构、预训练权重，设置图像预处理功能，排队快速读取输入。
#初始化后的主要工作流程是首先加载图像列表使用“enqueue\u image\u files”函数的文件，然后将其推送.通过网络与“feed_forward_batch”。



        self._network_name = network_name
        self._checkpoint_path = checkpoint_path
        self._batch_size = batch_size
        self._num_classes = num_classes

        self._preproc_func_name = preproc_func_name
        self._num_preproc_threads = preproc_threads

        self._global_step = tf.train.get_or_create_global_step()

        self._image_size = self._network_fn.default_image_size

        #  用文件名队列设置输入管道
        self._filename_queue = tf.FIFOQueue(100000, [tf.string], shapes=[[]], name="filename_queue")
        self._pl_excel_files = tf.placeholder(tf.string, shape=[None], name="excel_file_list")
        self._enqueue_op = self._filename_queue.enqueue_many([self._pl_excel_files])
        self._num_in_queue = self._filename_queue.size()

        # 图像阅读器和预处理器
        self._batch_from_queue, self._batch_filenames = \
            self._preproc_image_batch(self._batch_size, num_threads=preproc_threads)

        # 使用占位符作为输入或从队列中获取
        self._image_batch = tf.placeholder_with_default(
            self._batch_from_queue, shape=[None, self._image_size, self._image_size, 3])

        # 检索登录和网络端点（用于提取激活）
        # Note: endpoints is a dictionary with endpoints[name] = tf.Tensor
        self._logits, self._endpoints = self._network_fn(self._image_batch)

        #查找检查点文件
        checkpoint_path = self._checkpoint_path
        if tf.gfile.IsDirectory(self._checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(self._checkpoint_path)

        # 将预训练的权重加载到模型中
        variables_to_restore = slim.get_variables_to_restore()
        restore_fn = slim.assign_from_checkpoint_fn(
            self._checkpoint_path, variables_to_restore)

        # 开始会话并加载预训练的权重
        self._sess = tf.Session()
        restore_fn(self._sess)

        # 局部变量初始值设定项，队列等所需。
        self._sess.run(tf.local_variables_initializer())

        # Managing the queues and threads
        self._coord = tf.train.Coordinator()
        self._threads = tf.train.start_queue_runners(coord=self._coord, sess=self._sess)

    def _preproc_image_batch(self, batch_size, num_threads=1):
        '''
        此函数仅用于队列输入管道。它读取文件名从文件名队列中，解码图像，将其推送到预处理函数，然后使用tf.train.bath 产生批量。
        :param batch_size: int, batch size
        :param num_threads: int, number of input threads (default=1)
        :return: tf.Tensor, batch of pre-processed input images
        '''

        if ("resnet_v2" in self._network_name) and (self._preproc_func_name is None):
            raise ValueError("When using ResNet, please perform the pre-processing "
                            "function manually. See here for details: " 
                            "https://github.com/tensorflow/models/tree/master/slim")

        # 读取足部压力数据
        reader = tf.WholeFileReader()
        image_filename, image_raw = reader.read(self._filename_queue)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        # Image preprocessing
        preproc_func_name = self._network_name if self._preproc_func_name is None else self._preproc_func_name
       # image_preproc_fn = preprocessing_factory.get_preprocessing(preproc_func_name, is_training=False)
       # image_preproc = image_preproc_fn(image, self.image_size, self.image_size)
        # Read a batch of preprocessing images from queue
       # image_batch = tf.train.batch(
       #     [image_preproc, image_filename], batch_size, num_threads=num_threads,
        #    allow_smaller_final_batch=True)
        #return image_batch

    def enqueue_excel_files(self, image_files):
        '''
        Given a list of input images, feed these to the queue.
        :param image_files: list of str, list of image files to feed to filename queue
        '''
        self._sess.run(self._enqueue_op, feed_dict={self._pl_image_files: image_files})

    def feed_forward_batch(self, layer_names, images=None, fetch_images=False):
        '''
       通过网络推送一批图像的主要方法。有两个输入选项：（1）向图像提供图像文件名列表或（2）使用文件输入队列。确定使用哪种输入方法
是否指定了“images”参数。如果没有，那么队列使用。此函数返回一个输出字典，其中对应于图层名（以及“文件名”和“队列中的示例”）和张量值。
        :param layer_names: list of str, layer names to extract features from
        :param images: list of str, optional list of image filenames (default=None)
        :param fetch_images: bool, optionally fetch the input images (default=False)
        :return: dict, dictionary with values for all fetches
        '''

        # 要获取的网络（激活）字典
        fetches = {}

        # Check if all layers are available
        available_layers = self.layer_names()
        for layer_name in layer_names:
            if layer_name not in available_layers:
                raise ValueError("Unable to extract features for layer: {}".format(layer_name))
            fetches[layer_name] = self._endpoints[layer_name]

        # Manual inputs using placeholder 'images' of shape [N,H,W,C]
        feed_dict = None
        if images is not None:
            feed_dict = {self._image_batch: images}
        else:
            feed_dict = None
            fetches["filenames"] = self._batch_filenames

        # Optionally, we fetch the input image (for debugging/viz)
        if fetch_images:
            fetches["images"] = self._image_batch

        # Fetch how many examples left in queue
        fetches["examples_in_queue"] = self._num_in_queue

        # Actual forward pass through the network
        outputs = self._sess.run(fetches, feed_dict=feed_dict)
        return outputs

    def num_in_queue(self):

        # int, 返回队列中当前的示例数

        return self._sess.run(self._num_in_queue)

    def layer_names(self):
        '''
         str列表,网络中的层表名称
        '''
        return self._endpoints.keys()

    def layer_size(self, name):
        '''
        参数 name: str, 图层名
        :return: list of int, 图层名的形状
        '''
        return self._endpoints[name].get_shape().as_list()

    def print_network_summary(self):
        '''
        Prints the network layers and their shapes
        '''
        for name, tensor in self._endpoints.items():
            print("{} has shape {}".format(name, tensor.shape))

    def close(self):

        #停止预处理线程并关闭会话

        self._coord.request_stop()
        self._sess.run(self._filename_queue.close(cancel_pending_enqueues=True))
        self._coord.join(self._threads)
        self._sess.close()



    def batch_size(self):
        return self._batch_size


    def num_preproc_threads(self):
        return self._num_preproc_threads
