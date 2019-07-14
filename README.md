
# awesome-deep-learning

学习过程中，收集的深度学习资料，在不断更新中。 
如果您有好的学习资料，请联系我们，QQ群：791193818

## 目录
- [1 深度学习基础知识](https://github.com/DenseAI/awesome-deep-learning#1-深度学习基础知识)
	- [1.1 数学基础](https://github.com/DenseAI/awesome-deep-learning#11-数学基础)
	- [1.2 网络元素](https://github.com/DenseAI/awesome-deep-learning#12-网络元素)
	- [1.3 网络结构](https://github.com/DenseAI/awesome-deep-learning#13-网络结构)
	- [1.4 优化算法](https://github.com/DenseAI/awesome-deep-learning#14-优化算法)
	- [1.5 深度学习例子](https://github.com/DenseAI/awesome-deep-learning#15-深度学习例子)
- [2 目标检测](https://github.com/DenseAI/awesome-deep-learning#2-目标识别)	
	- [2.1 综述](https://github.com/DenseAI/awesome-deep-learning#21-综述)
	- [2.2 计算机视觉基础](https://github.com/DenseAI/awesome-deep-learning#22-计算机视觉基础)
	- [2.3 目标检测框架](https://github.com/DenseAI/awesome-deep-learning#23-目标检测框架)
	
## 1 深度学习基础知识 

#### 1.1 数学基础

包含深度神经网络、卷积神经网络、循环神经网络的前向传播、后向传播公式推导，以及损失函数和激活函数的推导，学习重点：前向、后向公式推导、激活函数、损失函数，特别是为什么使用交叉熵损失函数。

- [深度神经网络（DNN）模型与前向传播算法](https://www.cnblogs.com/pinard/p/6418668.html)
- [深度神经网络（DNN）反向传播算法(BP)](https://www.cnblogs.com/pinard/p/6422831.html)
- [深度神经网络（DNN）损失函数和激活函数的选择](https://www.cnblogs.com/pinard/p/6437495.html)
- [深度神经网络（DNN）的正则化](https://www.cnblogs.com/pinard/p/6472666.html)
- [卷积神经网络(CNN)模型结构](https://www.cnblogs.com/pinard/p/6483207.html)
- [卷积神经网络(CNN)前向传播算法](https://www.cnblogs.com/pinard/p/6489633.html)
- [卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html)
- [循环神经网络(RNN)模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6509630.html)
- [LSTM模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6519110.html)


#### 1.2 网络元素
- [多层感知机MLP](http://zh.d2l.ai/chapter_deep-learning-basics/mlp.html)

卷积神经网络，学习重点：核、步幅、填充、池化
- [卷积神经网络CNN](https://www.jianshu.com/p/70b6f5653ac6)
	- [二维卷积层](http://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)
	- [填充和步幅](http://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)
	- [多输入通道和多输出通道](http://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)
	- [池化层](http://zh.d2l.ai/chapter_convolutional-neural-networks/pooling.html)

循环神经网络，学习重点：BPTT、LSTM、GRU
- [循环神经网络RNN](https://www.jianshu.com/p/39a99c88a565)
	- [循环神经网络](http://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html)
	- [通过时间反向传播BPTT](http://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html)
	- [长短期记忆（LSTM）](http://zh.d2l.ai/chapter_recurrent-neural-networks/lstm.html)
	- [门控循环单元（GRU）](http://zh.d2l.ai/chapter_recurrent-neural-networks/gru.html)
	- [双向循环神经网络](http://zh.d2l.ai/chapter_recurrent-neural-networks/bi-rnn.html)
	
#### 1.3 网络结构

网络结构，学习重点：VGG、GooLeNet、ResNet、DPN

- [LeNet](https://blog.csdn.net/chenyuping333/article/details/82177677)
	- [LeNet-Keras](https://github.com/DustinAlandzes/mnist-lenet-keras/blob/master/lenet.py)
- [AlexNet](https://blog.csdn.net/chenyuping333/article/details/82178335)
	- [AlexNet-Keras](https://github.com/uestcsongtaoli/AlexNet/blob/master/model.py)
- [VGG](https://blog.csdn.net/chenyuping333/article/details/82250931)
	- [VGG16-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)
	- [VGG19-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)
- [GoogLeNet](https://blog.csdn.net/chenyuping333/article/details/82343608)
	- [GoogLeNet-Keras](https://github.com/dingchenwei/googLeNet/blob/master/googLeNet.py)
	- [inception_resnet_v2-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py)
	- [inception_v3-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py)
- [ResNet](https://blog.csdn.net/chenyuping333/article/details/82344334)
	- [resnet-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet.py)
	- [resnet_v2-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_v2.py)
- [DenseNet](https://blog.csdn.net/chenyuping333/article/details/82414542)
	- [DenseNet-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py)
- [ResNeXt](https://blog.csdn.net/chenyuping333/article/details/82453632)
	- [ResNext-Keras](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnext.py)
- [DPN Dual Path Net](https://blog.csdn.net/chenyuping333/article/details/82453965)
	- [DPN-Keras](https://github.com/titu1994/Keras-DualPathNetworks/blob/master/dual_path_network.py)
- [SeNet](http://www.sohu.com/a/161633191_465975)
	- [SeNet-Caffe](https://github.com/hujie-frank/SENet)

#### 1.4 优化算法

优化算法，学习重点：SGD、AdaGrad、RMSProp、Adam

- [随机梯度下降SGD](http://zh.d2l.ai/chapter_optimization/gd-sgd.html)
- [动量法](http://zh.d2l.ai/chapter_optimization/momentum.html)
- [AdaGrad算法](http://zh.d2l.ai/chapter_optimization/adagrad.html)
- [RMSProp算法](http://zh.d2l.ai/chapter_optimization/rmsprop.html)
- [AdaDelta算法](http://zh.d2l.ai/chapter_optimization/adadelta.html)
- [Adam算法](http://zh.d2l.ai/chapter_optimization/adam.html)

#### 1.5 深度学习例子

使用Keras进行文本分类、数字分类、图像分类，了解基本深度学习使用场景

- [使用卷积神经网络进行文本分类](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)
- [使用LSTM进行文本分类](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)
- [使用双向LSTM进行文本分类](https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py)
- [使用CNN+LSTM进行文本分类](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn_lstm.py)
- [使用CNN进行MNIST数字分类](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
- [使用CNN进行CIFAR10图像分类](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)
- [使用ResNet进行CIFAR10图像分类](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)


## 2 目标检测
从目标检查开始，逐渐结合论文、代码、应用等，难度慢慢增大了
#### 2.1 综述
中国国防科技大学、芬兰奥卢大学、澳大利亚悉尼大学、中国香港中文大学和加拿大滑铁卢大学等人推出一篇最新目标检测综述，详细阐述了当前目标检测最新成就和关键技术。
- [目标检测综述](https://blog.csdn.net/qq_35451572/article/details/82752261)
	- [Deep Learning for Generic Object Detection: A Survey](https://arxiv.org/pdf/1809.02165.pdf)
	
#### 2.2 计算机视觉基础
学习重点：图像增广、边界框、锚框、交并比IoU、非极大值抑制（NMS）
- [图像增广](http://zh.d2l.ai/chapter_computer-vision/image-augmentation.html)
- [微调](http://zh.d2l.ai/chapter_computer-vision/fine-tuning.html)
- [目标检测和边界框](http://zh.d2l.ai/chapter_computer-vision/bounding-box.html)
- [锚框](http://zh.d2l.ai/chapter_computer-vision/anchor.html)

#### 2.3 目标检测框架
<p align="center">
  <img width="1000" src="/assets/deep_learning_object_detection_history.png" "Example of object detection.">
</p>

- [一文读懂目标检测](https://cloud.tencent.com/developer/news/281788)
- **[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation | **[CVPR' 14]** |[`[pdf]`](https://arxiv.org/pdf/1311.2524.pdf) [`[official code - caffe]`](https://github.com/rbgirshick/rcnn)
- **[Fast R-CNN]** Fast R-CNN | **[ICCV' 15]** |[`[pdf]`](https://arxiv.org/pdf/1504.08083.pdf) [`[official code - caffe]`](https://github.com/rbgirshick/fast-rcnn) 
- **[Faster R-CNN, RPN]** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | **[NIPS' 15]** |[`[pdf]`](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  [`[official code - caffe]`](https://github.com/rbgirshick/py-faster-rcnn) [`[unofficial code - tensorflow]`](https://github.com/endernewton/tf-faster-rcnn)  [`[unofficial code - pytorch]`](https://github.com/jwyang/faster-rcnn.pytorch) 
- **[YOLO v1]** You Only Look Once: Unified, Real-Time Object Detection | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1506.02640.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) 
- **[SSD]** SSD: Single Shot MultiBox Detector | **[ECCV' 16]** |[`[pdf]`](https://arxiv.org/pdf/1512.02325.pdf) [`[official code - caffe]`](https://github.com/weiliu89/caffe/tree/ssd) [`[unofficial code - tensorflow]`](https://github.com/balancap/SSD-Tensorflow) [`[unofficial code - pytorch]`](https://github.com/amdegroot/ssd.pytorch) 
- **[YOLO v2]** YOLO9000: Better, Faster, Stronger | **[CVPR' 17]** |[`[pdf]`](https://arxiv.org/pdf/1612.08242.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) [`[unofficial code - caffe]`](https://github.com/quhezheng/caffe_yolo_v2) [`[unofficial code - tensorflow]`](https://github.com/nilboy/tensorflow-yolo) [`[unofficial code - tensorflow]`](https://github.com/sualab/object-detection-yolov2) [`[unofficial code - pytorch]`](https://github.com/longcw/yolo2-pytorch) 
- **[FPN]** Feature Pyramid Networks for Object Detection  | **[CVPR' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) [`[unofficial code - caffe]`](https://github.com/unsky/FPN)
- **[RetinaNet]** Focal Loss for Dense Object Detection | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1708.02002.pdf) [`[official code - keras]`](https://github.com/fizyr/keras-retinanet) [`[unofficial code - pytorch]`](https://github.com/kuangliu/pytorch-retinanet) [`[unofficial code - mxnet]`](https://github.com/unsky/RetinaNet) [`[unofficial code - tensorflow]`](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet)
- **[Mask R-CNN]** Mask R-CNN | **[ICCV' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) [`[official code - caffe2]`](https://github.com/facebookresearch/Detectron) [`[unofficial code - tensorflow]`](https://github.com/matterport/Mask_RCNN) [`[unofficial code - tensorflow]`](https://github.com/CharlesShang/FastMaskRCNN) [`[unofficial code - pytorch]`](https://github.com/multimodallearning/pytorch-mask-rcnn)
- **[YOLO v3]** YOLOv3: An Incremental Improvement | **[arXiv' 18]** |[`[pdf]`](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) [`[unofficial code - pytorch]`](https://github.com/ayooshkathuria/pytorch-yolo-v3) [`[unofficial code - pytorch]`](https://github.com/eriklindernoren/PyTorch-YOLOv3) [`[unofficial code - keras]`](https://github.com/qqwweee/keras-yolo3) [`[unofficial code - tensorflow]`](https://github.com/mystic123/tensorflow-yolo-v3)
- **[M2Det]** M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network | **[AAAI' 19]** |[`[pdf]`](https://arxiv.org/pdf/1811.04533.pdf) [`[official code - pytorch]`](https://github.com/qijiezhao/M2Det)
- [更多细节](https://github.com/hoya012/deep_learning_object_detection)
