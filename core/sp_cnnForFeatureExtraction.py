import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

"""
This file is a part of crnn-ctc-ocr project on GitHub
"""

def conv_feat_layers(inputs, width, training):
    #
    # convolutional features maps for recognition
    #

    #
    # recog-inputs should have shape [ batch, 36, width, channel]
    #
    # height_norm = 36
    #

    #
    # [3,1; 1,1],
    # [9,2; 3,2], [9,2; 3,2], [9,2; 3,2]
    # [18,4; 6,4], [18,4; 6,4], [18,4; 6,4]
    # [36,8; 12,8], [36,8; 12,8], [36,8; 12,8],
    #

    #
    layer_params = [[64, (3, 3), (1, 1), 'same', True, True, 'conv1'],
                    [64, (3, 3), (1, 1), 'same', True, True, 'conv2'],
                    [64, (2, 2), (2, 2), 'valid', True, True, 'pool1'],  # for pool
                    [128, (3, 3), (1, 1), 'same', True, True, 'conv3'],
                    [128, (3, 3), (1, 1), 'same', True, True, 'conv4'],
                    [128, (2, 2), (2, 2), 'valid', True, True, 'pool2'],  # for pool
                    [256, (3, 3), (1, 1), 'same', True, True, 'conv5'],
                    [256, (3, 3), (1, 1), 'same', True, True, 'conv6'],
                    [256, (3, 2), (3, 2), 'valid', True, True, 'pool3'],  # for pool
                    [512, (3, 1), (1, 1), 'valid', True, True, 'conv_feat']]  # for feat

    #
    with tf.variable_scope("conv_comm"):
        #
        inputs = conv_layer(inputs, layer_params[0], training)
        inputs = conv_layer(inputs, layer_params[1], training)
        inputs = padd_layer(inputs, [[0, 0], [0, 0], [0, 1], [0, 0]], name='padd1')
        inputs = conv_layer(inputs, layer_params[2], training)
        # inputs = layers.maxpool_layer(inputs, (2,2), (2,2), 'valid', 'pool1')
        #
        params = [[64, 3, (1, 1), 'same', True, True, 'conv1'],
                  [64, 3, (1, 1), 'same', True, False, 'conv2']]
        inputs = block_resnet_others(inputs, params, True, training, 'res1')
        #
        inputs = conv_layer(inputs, layer_params[3], training)
        inputs = conv_layer(inputs, layer_params[4], training)
        inputs = padd_layer(inputs, [[0, 0], [0, 0], [0, 1], [0, 0]], name='padd2')
        inputs = conv_layer(inputs, layer_params[5], training)
        # inputs = layers.maxpool_layer(inputs, (2,2), (2,2), 'valid', 'pool2')
        #
        params = [[128, 3, (1, 1), 'same', True, True, 'conv1'],
                  [128, 3, (1, 1), 'same', True, False, 'conv2']]
        inputs = block_resnet_others(inputs, params, True, training, 'res2')
        #
        inputs = conv_layer(inputs, layer_params[6], training)
        inputs = conv_layer(inputs, layer_params[7], training)
        inputs = padd_layer(inputs, [[0, 0], [0, 0], [0, 1], [0, 0]], name='padd3')
        inputs = conv_layer(inputs, layer_params[8], training)
        # inputs = layers.maxpool_layer(inputs, (3,2), (3,2), 'valid', 'pool3')
        #
        params = [[256, 3, (1, 1), 'same', True, True, 'conv1'],
                  [256, 3, (1, 1), 'same', True, False, 'conv2']]
        inputs = block_resnet_others(inputs, params, True, training, 'res3')
        #
        conv_feat = conv_layer(inputs, layer_params[9], training)
        #
    #
    # Calculate resulting sequence length from original image widths
    #
    two = tf.constant(2, dtype=tf.float32, name='two')
    #
    w = tf.cast(width, tf.float32)
    #
    w = tf.div(w, two)
    w = tf.ceil(w)
    #
    w = tf.div(w, two)
    w = tf.ceil(w)
    #
    w = tf.div(w, two)
    w = tf.ceil(w)
    #
    w = tf.cast(w, tf.int32)
    #
    # Vectorize
    sequence_length = tf.reshape(w, [-1], name='seq_len')
    #

    #
    return conv_feat, sequence_length
    #


def conv_layer(inputs, params, training):
    '''define a convolutional layer with params'''
    #
    # 输入数据维度为 4-D tensor: [batch_size, width, height, channels]
    #                         or [batch_size, height, width, channels]
    #
    # params = [filters, kernel_size, strides, padding, batch_norm, relu, name]
    #
    # batch_norm = True or False
    # relu = True or False
    #
    #
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    # kernel_initializer = tf.contrib.layers.xavier_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    outputs = tf.layers.conv2d(inputs, params[0], params[1], strides=params[2],
                               padding=params[3],
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               name=params[6])
    #
    if params[4]:  # batch_norm
        outputs = norm_layer(outputs, training, name=params[6] + '/batch_norm')
    #
    if params[5]:  # relu
        outputs = tf.nn.relu(outputs, name=params[6] + '/relu')
    #
    return outputs
    #


def norm_layer(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    #
    with tf.variable_scope(name, default_name='batch_norm'):
        #
        params_shape = [x.shape[-1]]  #
        batch_dims = list(range(0, len(x.shape) - 1))  #
        #
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer(),
                                          trainable=False)

        #
        def mean_var_with_update():
            #
            # axis = list(np.arange(len(x.shape) - 1))
            batch_mean, batch_variance = tf.nn.moments(x, batch_dims, name='moments')
            #
            with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, decay),
                                          assign_moving_average(moving_variance, batch_variance, decay)]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        #
        # mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        #
        if train:
            mean, variance = mean_var_with_update()
        else:
            mean, variance = moving_mean, moving_variance
            #
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer(),
                                   trainable=True)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=True)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        #
        return x
        #


'''
# tf.pad(tensor, paddings, mode='CONSTANT', name=None)
#
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1,], [2, 2]].
# rank of 't' is 2.
#
# padd1 = padd_layer(conv1, [[0,0],[0,0],[0,1],[0,0]], name='padd1')
'''


def padd_layer(tensor, paddings, mode='CONSTANT', name=None):
    ''' define padding layer '''
    return tf.pad(tensor, paddings, mode, name)
    #


'''
tf.layers.max_pooling2d(inputs, pool_size, strides,
                        padding='valid', data_format='channels_last', name=None)
'''


# 最大采提层
def maxpool_layer(inputs, size, stride, padding, name):
    '''define a max-pooling layer'''
    return tf.layers.max_pooling2d(inputs, size, stride,
                                   padding=padding,
                                   name=name)
    #


'''
tf.layers.average_pooling2d(inputs, pool_size, strides,
                            padding='valid', data_format='channels_last', name=None)
'''


# 均值采提层
def averpool_layer(inputs, size, stride, padding, name):
    '''define a average-pooling layer'''
    return tf.layers.average_pooling2d(inputs, size, stride,
                                       padding=padding,
                                       name=name)
    #


'''
fc = tf.layers.dense(rnn2, fc_size,
                     activation = tf.nn.relu,
                     kernel_initializer = weight_initializer,
                     bias_initializer = bias_initializer,
                     name = 'fc')
#
# dense operates on the last dim
#
# activation = tf.nn.sigmoid,
# activation = tf.nn.tanh,
# activation = tf.nn.relu,
#

'''

'''
blocks

'''


def block_resnet_others(inputs, layer_params, relu, training, name):
    '''define resnet block'''
    #
    # 1，图像大小不缩小，或者，图像大小只能降，1/2, 1/3, 1/4, ...
    # 2，深度，卷积修改
    #
    with tf.variable_scope(name):
        #
        # short_cut = tf.add(inputs, 0)
        short_cut = tf.identity(inputs)
        #
        shape_in = inputs.get_shape().as_list()
        #
        for item in layer_params:
            inputs = conv_layer(inputs, item, training)
        #
        shape_out = inputs.get_shape().as_list()
        #
        # 图片大小，缩小
        if shape_in[1] != shape_out[1] or shape_in[2] != shape_out[2]:
            #
            size = [shape_in[1] // shape_out[1], shape_in[2] // shape_out[2]]
            #
            short_cut = maxpool_layer(short_cut, size, size, 'valid', 'shortcut_pool')
            #
        #
        # 深度
        if shape_in[3] != shape_out[3]:
            #
            item = [shape_out[3], 1, (1, 1), 'same', True, False, 'shortcut_conv']
            #
            short_cut = conv_layer(short_cut, item, training)
            #
        #
        outputs = tf.add(inputs, short_cut, name='add')
        #
        if relu: outputs = tf.nn.relu(outputs, 'last_relu')
        #
    #
    return outputs
    #


def block_resnet(inputs, filters, flag_size, relu, training, name):
    '''define resnet block'''
    #
    with tf.variable_scope(name):
        #
        if flag_size == 1:  # same_size
            #
            item1 = [filters, (3, 3), (1, 1), 'same', True, True, 'conv1']
            item2 = [filters, (3, 3), (1, 1), 'same', True, False, 'conv2']
            outputs = conv_layer(inputs, item1, training)
            outputs = conv_layer(outputs, item2, training)
            #
            outputs = tf.add(outputs, inputs, name='add')
            if relu: outputs = tf.nn.relu(outputs, 'last_relu')
            #
            return outputs
            #
        elif flag_size == 2:  # half_size
            #
            outputs = padd_layer(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], name='padd')
            #
            item1 = [filters, (3, 3), (2, 2), 'valid', True, True, 'conv1']
            item2 = [filters, (3, 3), (1, 1), 'same', True, False, 'conv2']
            outputs = conv_layer(outputs, item1, training)
            outputs = conv_layer(outputs, item2, training)
            #
            short_cut = maxpool_layer(inputs, (2, 2), (2, 2), 'valid', 'skip_pool')
            #
            item = [filters, 1, (1, 1), 'same', True, False, 'skip_conv']
            short_cut = conv_layer(short_cut, item, training)
            #
            outputs = tf.add(outputs, short_cut, name='add')
            if relu: outputs = tf.nn.relu(outputs, 'last_relu')
            #
            return outputs
            #
        else:
            print('flag_size not 1 or 2, in block_resnet_paper()')
            #
            return inputs
            #


def block_bottleneck(inputs, depth_arr, relu, training, name):
    '''define bottleneck block'''
    #
    # shape_in = inputs.get_shape().as_list()
    #
    # short_cut = inputs
    #
    with tf.variable_scope(name):
        #
        item1 = [depth_arr[0], (1, 1), (1, 1), 'same', True, True, 'conv1']
        item2 = [depth_arr[1], (3, 3), (1, 1), 'same', True, True, 'conv2']
        item3 = [depth_arr[2], (1, 1), (1, 1), 'same', True, False, 'conv3']
        #
        outputs = conv_layer(inputs, item1, training)
        outputs = conv_layer(outputs, item2, training)
        outputs = conv_layer(outputs, item3, training)
        #
        outputs = tf.add(outputs, inputs, name='add')
        if relu: outputs = tf.nn.relu(outputs, 'last_relu')
        #
    #
    return outputs
    #


def block_inception(inputs, K, depth_arr, relu, training, name):
    ''' define inception-like block '''
    #
    with tf.variable_scope(name):
        #
        params_1 = [depth_arr[0], [1, K], (1, 1), 'same', True, False, 'branch1']
        params_2 = [depth_arr[1], [K, 1], (1, 1), 'same', True, False, 'branch2']
        params_3_1 = [depth_arr[2], [1, K], (1, 1), 'same', True, False, 'branch3_1']
        params_3_2 = [depth_arr[3], [K, 1], (1, 1), 'same', True, False, 'branch3_2']
        params_4 = [depth_arr[4], [K, K], (1, 1), 'same', True, False, 'branch4']
        #
        branch_1 = conv_layer(inputs, params_1, training)
        branch_2 = conv_layer(inputs, params_2, training)
        branch_3 = conv_layer(inputs, params_3_1, training)
        branch_3 = conv_layer(branch_3, params_3_2, training)
        branch_4 = conv_layer(inputs, params_4, training)
        #
        outputs = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
        #
        if relu: outputs = tf.nn.relu(outputs, 'last_relu')
        #
    #
    return outputs
    #

