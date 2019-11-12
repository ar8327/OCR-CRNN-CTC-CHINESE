import tensorflow as tf
import core.custom_layers as cLayers
from core.sp_cnnForFeatureExtraction import *
from core.custom_layers import *
from core.meta import *
import core.meta as meta
import os
def ModelTrain():
    graph = tf.Graph()
    with graph.as_default():
        #
        x = tf.placeholder(tf.float32, (None, None, None, 3), name='x-input')
        w = tf.placeholder(tf.int32, (None,), name='w-input')  # width
        y = tf.sparse_placeholder(tf.int32, name='y-input')
        #
        conv_feat, seq_len = conv_feat_layers(x, w, training=True)  # train
        result_logits = rnn_recog_layers(conv_feat, seq_len, len(alphabet) + 1)
        result_decoded_list = decode_rnn_results_ctc_beam(result_logits, seq_len)
        loss = ctc_loss_layer(y, result_logits, seq_len)
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=MOMENTUM)
        grads_applying = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_applying, global_step=global_step)

        return graph,y,train_op #Return y & train_op to fix no op found error


def ModelPredict():
    graph = tf.Graph()
    with graph.as_default():
        #
        x = tf.placeholder(tf.float32, (None, None, None, 3), name='x-input')
        w = tf.placeholder(tf.int32, (None,), name='w-input')  # width
        y = tf.sparse_placeholder(tf.int32, name='y-input')
        #
        conv_feat, seq_len = conv_feat_layers(x, w, training=False)  # train
        result_logits = rnn_recog_layers(conv_feat, seq_len, len(alphabet) + 1)
        result_decoded_list = decode_rnn_results_ctc_beam(result_logits, seq_len)
        loss = ctc_loss_layer(y, result_logits, seq_len)
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=MOMENTUM)
        grads_applying = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_applying, global_step=global_step)

        return graph #Return y & train_op to fix no op found error
