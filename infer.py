# -*- coding: utf-8 -*-
# LeNet-5 Model
import tensorflow as tf 

# SETTINGS
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_LABELS = 17

CONV1_KSIZE = 3
CONV1_DEEP = 64
CONV2_KSIZE = 3
CONV2_DEEP = 128
CONV3_KSIZE = 3
CONV3_DEEP = 128
CONV4_KSIZE = 3
CONV4_DEEP = 64
CONV5_KSIZE = 3
CONV5_DEEP = 64
FC_SIZE = 128

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        name = "weights",
        shape = shape,
        initializer = tf.truncated_normal_initializer(stddev = 0.1))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights 

def inference(input_tensor, train, regularizer): 
    with tf.variable_scope("layer1_conv1"):
        # 声明权重
        conv1_weights = tf.get_variable(
            name="weight", 
            shape=[CONV1_KSIZE, CONV1_KSIZE, NUM_CHANNELS, CONV1_DEEP], 
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            name="bias",
            shape=[CONV1_DEEP],
            initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为32的kernel，
        conv1 = tf.nn.conv2d(
            input = input_tensor,
            filter = conv1_weights,
            strides = [1, 1, 1, 1],
            padding = "SAME")
        # print "conv1.shape: ", conv1.shape 
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        # print "relu1.shape: ", relu1.shape 
    # 声明pooling层
    with tf.variable_scope("layer2_pool1"):
        pool1 = tf.nn.max_pool(
            value = relu1, 
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = "SAME")
        # print "pool1.shape: ", pool1.shape 
    # 声明第二个卷积层的参数，并实现前向传播过程
    with tf.variable_scope("layer3_conv2"):
        # 声明权重
        conv2_weights = tf.get_variable(
            name = "weight",
            shape=[CONV2_KSIZE, CONV2_KSIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        # bias
        conv2_biases = tf.get_variable(
            name = "bias",
            shape=[CONV2_DEEP],
            initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
            input = pool1,
            filter = conv2_weights,
            strides = [1, 1, 1, 1],
            padding = "SAME")
        # print "conv2.shape: ", conv2.shape 
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        # print "relu2.shape: ", relu2.shape  
    
    # pool2
    with tf.variable_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(value = relu2, 
            ksize=[1, 2, 2, 1], 
            strides=[1, 2, 2, 1], 
            padding="SAME")
        # print "pool2.shape: ", pool2.shape
    
    # conv3
    with tf.variable_scope("layer5_conv3"):
        conv3_weights = tf.get_variable(
            name = "weight",
            shape = [CONV3_KSIZE, CONV3_KSIZE, CONV2_DEEP, CONV3_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv3_biases = tf.get_variable(
            name = "bias",
            shape = [CONV3_DEEP],
            initializer = tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(
            input = pool2,
            filter = conv3_weights,
            strides = [1, 1, 1, 1],
            padding = "SAME")
        # print "conv3.shape:", conv3.shape
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        # print "relu3.shape:", relu3.shape 
    # pool3
    with tf.variable_scope("layer6_pool3"):
        pool3 = tf.nn.max_pool(value = relu3, 
            ksize=[1, 2, 2, 1], 
            strides=[1, 2, 2, 1], 
            padding="SAME")
        # print "pool3.shape: ", pool3.shape

    # conv4
    with tf.variable_scope("layer7_conv4"):
        conv4_weights = tf.get_variable(
            name = "weight",
            shape = [CONV4_KSIZE, CONV4_KSIZE, CONV3_DEEP, CONV4_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv4_biases = tf.get_variable(
            name = "bias",
            shape = [CONV4_DEEP],
            initializer = tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(
            input = pool3,
            filter = conv4_weights,
            strides = [1, 1, 1, 1],
            padding = "SAME")
        # print "conv4.shape:", conv4.shape
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        # print "relu4.shape:", relu4.shape 
    # pool4
    with tf.variable_scope("layer8_pool4"):
        pool4 = tf.nn.max_pool(value = relu4,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], 
            padding="SAME")
        # print "pool4.shape: ", pool4.shape 
    # conv5      
    with tf.variable_scope("layer9_conv5"):
        conv5_weights = tf.get_variable(
            name = "weight",
            shape = [CONV5_KSIZE, CONV5_KSIZE, CONV4_DEEP, CONV5_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv5_biases = tf.get_variable(
            name = "bias",
            shape = [CONV5_DEEP],
            initializer = tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(
            input = pool4,
            filter = conv5_weights,
            strides = [1, 1, 1, 1],
            padding = "SAME")
        # print "conv5.shape:", conv5.shape
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
        # print "relu5.shape:", relu5.shape 
    # pool5
    with tf.variable_scope("layer10_pool4"):
        pool5 = tf.nn.max_pool(value = relu5,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], 
            padding="SAME")
        # print "pool5.shape: ", pool5.shape       
    # 获取最后一层卷积层的输出的维度
    poolled_shape = pool5.get_shape().as_list()
    # print "poolled_shape:", poolled_shape
    nodes = poolled_shape[1] * poolled_shape[2] * poolled_shape[3]
    # 将pool2的输出有shape = [N, H, W, C], reshape成[N, H*W*C]
    reshaped = tf.reshape(pool5,
            [poolled_shape[0], nodes])
    # fc1
    with tf.variable_scope("layer11_fc1"):
        # define weihts
        fc1_weights = tf.get_variable(
            name = "weight",
            shape=[nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 对全连接层的参数进行正则化
        if regularizer != None:
            tf.add_to_collection(name = "losses", value = regularizer(fc1_weights))
        # biases
        fc1_biases = tf.get_variable(
            name = "bias",
            shape=[FC_SIZE],
            initializer=tf.constant_initializer(0.1))
        # 前向传播
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 加入dropout
        if train:
            fc1 = tf.nn.dropout(
                x = fc1, 
                keep_prob = 1.0)
    # fc2
    with tf.variable_scope("layer12_fc2"):
        fc2_weights = tf.get_variable(
            name = "weights",
            shape = [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev = 0.1))
        fc2_biases = tf.get_variable(
            name = "biases",
            shape = [NUM_LABELS],
            initializer=tf.constant_initializer(0.1))
        # 
        if regularizer != None:
            tf.add_to_collection(name="losses", value = regularizer(fc2_weights))
        logits = tf.matmul(fc1, fc2_weights) + fc2_biases
    # 返回最后的输出
    return logits

    
