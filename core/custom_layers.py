import tensorflow as tf

def ctc_loss_layer(sequence_labels, rnn_logits, sequence_length): #CTC Loss
    #
    loss = tf.nn.ctc_loss(inputs = rnn_logits,
                          labels = sequence_labels,
                          sequence_length = sequence_length,
                          ignore_longer_outputs_than_inputs = True,
                          time_major = True )
    #
    total_loss = tf.reduce_mean(loss, name = 'loss')
    #
    return total_loss
    #


def decode_rnn_results_ctc_beam(results, seq_len,beam_width=1,top_paths=1): #CTC Beam Searcher
    #
    # tf.nn.ctc_beam_search_decoder
    #
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(results, seq_len, merge_repeated=False,beam_width=beam_width,top_paths=top_paths)
    #
    return decoded
    #

def bi_gru(input_sequence, sequence_length, rnn_size, scope):
    cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
    cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length=sequence_length,
                                                    time_major=True,
                                                    dtype=tf.float32,
                                                    scope=scope)

    rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')
    return rnn_output_stack


def rnn_recog_layers(features, sequence_length, num_classes): #Rnn decoder with fn as logits
    #
    # batch-picture features
    features = tf.squeeze(features, axis=1)  # squeeze
    #
    # [batchSize paddedSeqLen numFeatures]
    #
    #
    rnn_size = 256  # 256, 512
    # fc_size = 512  # 256, 384, 512
    #
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    #
    # Transpose to time-major order for efficiency
    #  --> [paddedSeqLen batchSize numFeatures]
    #
    rnn_sequence = tf.transpose(features, perm=[1, 0, 2], name='time_major')
    #
    rnn1 = bi_gru(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
    rnn2 = bi_gru(rnn1, sequence_length, rnn_size, 'bdrnn2')
    #
    # out
    #
    rnn_logits = tf.layers.dense(rnn2, num_classes,
                                 activation=None,  # tf.nn.sigmoid,
                                 kernel_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 name='rnn_logits')
    #
    # dense operates on last dim
    #

    #
    return rnn_logits
    #
