
# awesome-medical-machine-learning

学习过程中，收集的深度学习资料，在不断更新中。。。 
如果您有好的学习资料，请联系我们，QQ群：791193818

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
- [随机梯度下降SGD](http://zh.d2l.ai/chapter_optimization/gd-sgd.html)
- [动量法](http://zh.d2l.ai/chapter_optimization/momentum.html)
- [AdaGrad算法](http://zh.d2l.ai/chapter_optimization/adagrad.html)
- [RMSProp算法](http://zh.d2l.ai/chapter_optimization/rmsprop.html)
- [AdaDelta算法](http://zh.d2l.ai/chapter_optimization/adadelta.html)
- [Adam算法](http://zh.d2l.ai/chapter_optimization/adam.html)

