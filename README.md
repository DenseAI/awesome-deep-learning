
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
	- [2.4 代码详解](https://github.com/DenseAI/awesome-deep-learning#24-代码详解)
	- [2.5 人脸识别](https://github.com/DenseAI/awesome-deep-learning#25-人脸识别)
- [3 强化学习](https://github.com/DenseAI/awesome-deep-learning#3-强化学习)	
	- [3.1 基础知识](https://github.com/DenseAI/awesome-deep-learning#31-基础知识)
	- [3.2 强化学习基础](https://github.com/DenseAI/awesome-deep-learning#32-强化学习基础)
	- [3.3 强化学习与Python](https://github.com/DenseAI/awesome-deep-learning#33-强化学习与python)
	- [3.4 AlphaGo Zero](https://github.com/DenseAI/awesome-deep-learning#34-alphago-zero)
- [4 生成对抗网络（GAN）](https://github.com/DenseAI/awesome-deep-learning#4-生成对抗网络gan)	
	- [4.1 综述](https://github.com/DenseAI/awesome-deep-learning#41-综述)
	- [4.2 各种类型的GAN](https://github.com/DenseAI/awesome-deep-learning#42-各种类型的gan)
	- [4.3 生成模型](https://github.com/DenseAI/awesome-deep-learning#43-生成模型)
- [5 自然语言处理(NLP)](https://github.com/DenseAI/awesome-deep-learning#5-自然语言处理nlp)
	- [5.1 词向量（Word2vec）](https://github.com/DenseAI/awesome-deep-learning#51-词向量word2vec)
	
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
- [论文笔记-2019-Object Detection in 20 Years: A Survey](https://blog.csdn.net/clover_my/article/details/92794719)
	- [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055v2)
- [人脸识别综述](https://www.cnblogs.com/shouhuxianjian/p/9789243.html)
	- [Deep Face Recognition: A Survey](https://arxiv.org/pdf/1804.06655)
	
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
#### 2.4 代码详解
- [YOLO v3]
	- [探索 YOLO v3 源码 - 第1篇 训练](https://mp.weixin.qq.com/s/T9LshbXoervdJDBuP564dQ)
	- [探索 YOLO v3 源码 - 第2篇 模型](https://mp.weixin.qq.com/s/N79S9Qf1OgKsQ0VU5QvuHg)
	- [探索 YOLO v3 源码 - 第3篇 网络](https://mp.weixin.qq.com/s/hC4P7iRGv5JSvvPe-ri_8g)
	- [探索 YOLO v3 源码 - 第4篇 真值](https://mp.weixin.qq.com/s/5Sj7QadfVvx-5W9Cr4d3Yw)
	- [探索 YOLO v3 源码 - 第5篇 Loss](https://mp.weixin.qq.com/s/4L9E4WGSh0hzlD303036bQ)
	- [探索 YOLO v3 源码 - 完结篇 预测](https://mp.weixin.qq.com/s/J1ddmUvT_F2HcljLtg_uWQ)
- [目标检测算法综述之FPN优化篇](https://zhuanlan.zhihu.com/p/63047557)
- [FaceNet源码解读1](https://blog.csdn.net/u013044310/article/details/79556099)
- [FaceNet源码解读2](https://blog.csdn.net/u013044310/article/details/80481642)
- [格灵深瞳：人脸识别最新进展以及工业级大规模人脸识别实践探讨](https://blog.csdn.net/guleileo/article/details/80863579)	
- [DeepID+DeepID2+DeepID3人脸识别算法总结](https://blog.csdn.net/weixin_42546496/article/details/88537882)	
- [insightface](https://github.com/deepinsight/insightface)	
#### 2.5 人脸识别
- [人脸识别系列（一）:DeepFace ](https://blog.csdn.net/Fire_Light_/article/details/79558759)
- [人脸识别系列（二）：DeepID1 ](https://blog.csdn.net/Fire_Light_/article/details/79559312)
- [人脸识别系列（三）：DeepID2 ](https://blog.csdn.net/Fire_Light_/article/details/79559051)
- [人脸识别系列（四）：Webface系列1（CASIA-WebFace）](https://blog.csdn.net/Fire_Light_/article/details/79588429)
- [人脸识别系列（五）：face++ ](https://blog.csdn.net/Fire_Light_/article/details/79590811)
- [人脸识别系列（六）：FaceNet ](https://blog.csdn.net/Fire_Light_/article/details/79592804)
- [人脸识别系列(七)：百度的人脸识别 ](https://blog.csdn.net/Fire_Light_/article/details/79589926)
- [人脸识别系列（八）：VGGFace ](https://blog.csdn.net/Fire_Light_/article/details/79593778)
- [人脸识别系列（九）：FR+FCN ](https://blog.csdn.net/Fire_Light_/article/details/79594590)
- [人脸识别系列（十）：Webface系列2 ](https://blog.csdn.net/Fire_Light_/article/details/79595687)
- [人脸识别系列（十一）：Webface系列3](https://blog.csdn.net/Fire_Light_/article/details/79596024)
- [人脸识别系列（十二）：Center Loss ](https://blog.csdn.net/Fire_Light_/article/details/79598497)
- [人脸识别系列（十三）：SphereFace ](https://blog.csdn.net/Fire_Light_/article/details/79599020)
- [人脸识别系列（十四）：NormFace ](https://blog.csdn.net/Fire_Light_/article/details/79601378)
- [人脸识别系列（十五）：COCO Loss ](https://blog.csdn.net/Fire_Light_/article/details/79602134)
- [人脸识别系列（十六）：AMSoftmax ](https://blog.csdn.net/Fire_Light_/article/details/79602310)
- [人脸识别系列（十七）：ArcFace/Insight Face ](https://blog.csdn.net/Fire_Light_/article/details/79602705)
- [人脸识别系列（十八）：MobileFaceNets ](https://blog.csdn.net/Fire_Light_/article/details/80279342)



## 3 强化学习

#### 3.1 基础知识
- [模型基础](https://www.cnblogs.com/pinard/p/9385570.html)
- [马尔科夫决策过程(MDP)](https://www.cnblogs.com/pinard/p/9426283.html)
- [用动态规划（DP）求解](https://www.cnblogs.com/pinard/p/9463815.html)
- [用蒙特卡罗法（MC）求解](https://www.cnblogs.com/pinard/p/9492980.html)
- [基于模拟的搜索与蒙特卡罗树搜索(MCTS)](https://www.cnblogs.com/pinard/p/10470571.html)

#### 3.2 强化学习基础
- [时序差分离线控制算法Q-Learning](https://www.cnblogs.com/pinard/p/9669263.html)
- [时序差分在线控制算法SARSA](https://www.cnblogs.com/pinard/p/9614290.html)
- [价值函数的近似表示与Deep Q-Learning](https://www.cnblogs.com/pinard/p/9714655.html)
- [Deep Q-Learning进阶之Nature DQN](https://www.cnblogs.com/pinard/p/9756075.html)
- [策略梯度(Policy Gradient)](https://www.cnblogs.com/pinard/p/10137696.html)
- [Actor-Critic](https://www.cnblogs.com/pinard/p/10272023.html)
- [A3C](https://www.cnblogs.com/pinard/p/10334127.html)
- [深度确定性策略梯度(DDPG)](https://www.cnblogs.com/pinard/p/10345762.html)

#### 3.3 强化学习与Python
上面3.2 强化学习基础，包含很多公式，可能看起来有点吃力，下面是莫烦的强化学习知识点，可能稍微好一点，看起来不这么费劲。
- [Q-learning]
	- [什么是 Q Leaning](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-A-q-learning/)
	- [小例子](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/)
	- [Q-learning 算法更新](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-2-tabular-q1/)
	- [Q-learning 思维决策](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-3-tabular-q2/)
- [Sarsa]
	- [什么是 Sarsa](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/3-1-A-sarsa/)
	- [Sarsa 算法更新](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/3-1-tabular-sarsa1/)
	- [Sarsa 思维决策](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/3-2-tabular-sarsa2/)
	- [什么是 Sarsa(lambda)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/3-3-A-sarsa-lambda/)
	- [Sarsa-lambda](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/3-3-tabular-sarsa-lambda/)
- [Deep Q Network]
	- [什么是 DQN ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-A-DQN/)
	- [DQN 算法更新 (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-DQN1/)
	- [DQN 神经网络 (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-2-DQN2/)
	- [DQN 思维决策 (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/)
	- [OpenAI gym 环境库 ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-4-gym/)
	- [Double DQN (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-5-double_DQN/)
	- [Prioritized Experience Replay (DQN) (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/)
	- [Dueling DQN (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-7-dueling-DQN/)
- [Policy Gradient]
	- [什么是 Policy Gradients ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-A-PG/)
	- [Policy Gradients 算法更新 (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-policy-gradient-softmax1/)
	- [Policy Gradients 思维决策 (Tensorflow) ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-2-policy-gradient-softmax2/)
- [Actor Critic]
	- [什么是 Actor Critic ](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-A-AC/)
	- [Actor Critic (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-actor-critic/)
	- [什么是 Deep Deterministic Policy Gradient (DDPG)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-A-DDPG/)
	- [Deep Deterministic Policy Gradient (DDPG) (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-DDPG/)
	- [什么是 Asynchronous Advantage Actor-Critic (A3C)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-3-A1-A3C/)
	- [Asynchronous Advantage Actor-Critic (A3C) (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-3-A3C/)
	- [Distributed Proximal Policy Optimization (DPPO) (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-4-DPPO/)

#### 3.4 AlphaGo Zero	
- [AlphaGo Zero 详解](https://blog.csdn.net/windowsyun/article/details/88701321)
- [蒙特卡洛树搜索（MCTS）代码详解](https://blog.csdn.net/windowsyun/article/details/88770799)
- [AlphaZero五子棋网络模型](https://blog.csdn.net/windowsyun/article/details/88855277)


## 4 生成对抗网络（GAN）
#### 4.1 综述
- [万字综述之生成对抗网络（GAN）](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247495668&idx=1&sn=e7e959b2bdd7b2763b9207ccb80fa6bc&chksm=96ea3074a19db96208a51d26f7b5b4ef9c3a37a7799ec270becc77203de4294235041ede7206&token=2004841509&lang=zh_CN#rd)
	- [How Generative Adversarial Networks and Their Variants Work: An Overview](https://arxiv.org/abs/1711.05914)
	- [令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)

#### 4.2 各种类型的GAN
- [Generative Adversarial Network](https://github.com/eriklindernoren/Keras-GAN#gan)
- [Deep Convolutional GAN](https://github.com/eriklindernoren/Keras-GAN#dcgan)
- [Auxiliary Classifier GAN](https://github.com/eriklindernoren/Keras-GAN#ac-gan)
- [Wasserstein GAN](https://github.com/eriklindernoren/Keras-GAN#wgan)
- [Wasserstein GAN GP](https://github.com/eriklindernoren/Keras-GAN#wgan-gp)
- [CycleGAN](https://github.com/eriklindernoren/Keras-GAN#cyclegan)
- [Pix2Pix](https://github.com/eriklindernoren/Keras-GAN#pix2pix)
- [InfoGAN](https://github.com/eriklindernoren/Keras-GAN#infogan)

#### 4.3 生成模型
- [VAE(Variational Autoencoder)的原理](https://www.cnblogs.com/huangshiyu13/p/6209016.html)
- [变分自编码器（一）：原来是这么一回事 ](https://spaces.ac.cn/archives/5253)
- [变分自编码器（二）：从贝叶斯观点出发 ](https://spaces.ac.cn/archives/5343)
- [变分自编码器（三）：这样做为什么能成？ ](https://spaces.ac.cn/archives/5383)
- [变分自编码器（四）：一步到位的聚类方案 ](https://spaces.ac.cn/archives/5887)
- [VAE(Variational Autoencoder)的原理](https://www.cnblogs.com/huangshiyu13/p/6209016.html)

- [VQ-VAE的简明介绍：量子化自编码器](https://spaces.ac.cn/archives/6760/comment-page-1?replyTo=11553)

## 5 自然语言处理（NLP）

#### 5.1 词向量（Word2vec）
- [word2vec原理(一) CBOW与Skip-Gram模型基础](https://www.cnblogs.com/pinard/p/7160330.html)
- [word2vec原理(二) 基于Hierarchical Softmax的模型](https://www.cnblogs.com/pinard/p/7243513.html)
- [word2vec原理(三) 基于Negative Sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html)
- [NLP︱高级词向量表达（二）——FastText（简述、学习笔记）](https://blog.csdn.net/sinat_26917383/article/details/54850933)
	- [facebookresearch/fastText ](https://github.com/facebookresearch/fastText)
	- [ivanhk/fastText_java ](https://github.com/ivanhk/fastText_java)

