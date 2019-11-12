import tensorflow as tf
import core.meta as meta
import numpy as np
import PIL.Image as Image
import core.model as model
import core.custom_layers

def convert2SparseTensorValue(list_labels):
    #
    # list_labels: batch_major
    #

    #
    num_samples = len(list_labels)
    num_maxlen = max(map(lambda x: len(x), list_labels))
    #
    indices = []
    values = []
    shape = [num_samples, num_maxlen]
    #
    for idx in range(num_samples):
        #
        item = list_labels[idx]
        #
        values.extend(item)
        indices.extend([[idx, posi] for posi in range(len(item))])
        #
    #
    return tf.SparseTensorValue(indices=indices, values=values, dense_shape=shape)
    #


#
def convert2ListLabels(sparse_tensor_value):
    #
    # list_labels: batch_major
    #

    shape = sparse_tensor_value.dense_shape
    indices = sparse_tensor_value.indices
    values = sparse_tensor_value.values

    list_labels = []
    #
    item = [0] * shape[1]
    for i in range(shape[0]): list_labels.append(item)
    #

    for idx, value in enumerate(values):
        #
        posi = indices[idx]
        #
        list_labels[posi[0]][posi[1]] = value
        #

    return list_labels
    #
def predict(graph=None,image_in=None):
    if graph == None:
        graph = model.ModelPredict()

    if isinstance(image_in, str):
        img = Image.open(image_in)
        img = img.convert('RGB')
        img_size = img.size
        if img_size[1] != meta.height_norm:
            w = int(img_size[0] * meta.height_norm *1.0/img_size[1])
            img = img.resize((w, meta.height_norm))
        img_data = np.array(img, dtype = np.float32)/255  # (height, width, channel)
        img_data = [ img_data[:,:,0:3] ]
    else:
        img = image_in
        img_size = (image_in.shape[1], image_in.shape[0])
        if img_size[1] != meta.height_norm:
            w = int(img_size[0] * meta.height_norm * 1.0 / img_size[1])
            img = cv2.resize(img, (w, meta.height_norm), 0, 0)
        img_data = np.array(img, dtype=np.float32) / 255  # (height, width, channel)
        img_data = [img_data[:, :, 0:3]]
        img_data = img_data

    w_arr = [img_data[0].shape[1]]  # batch, height, width, channel
    with tf.Session(graph=graph) as sess:
        var_list = tf.trainable_variables()
        # for g in var_list:
        #     print(g.name)
        # print("===============================")
        g_list = tf.global_variables()
        # for g in g_list:
        #     print(g.name)
        bn_moving_vars = [g for g in g_list if 'mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
        # restore with saved data
        ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
        #
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        x = graph.get_tensor_by_name('x-input:0')
        w = graph.get_tensor_by_name('w-input:0')
        seq_len = graph.get_tensor_by_name('seq_len:0')
        result_logits = graph.get_tensor_by_name('rnn_logits/BiasAdd:0')
        result_i = graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
        result_v = graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
        result_s = graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
        feed_dict = {x: img_data, w: w_arr}
        #
        results, seq_length, d_i, d_v, d_s = \
        sess.run([result_logits, seq_len,
                  result_i, result_v, result_s], feed_dict)
        #

        # decoded = core.custom_layers.decode_rnn_results_ctc_beam(result_logits, seq_len)
        #
        # d_i = decoded[0].indices
        # d_v = decoded[0].values
        # d_s = decoded[0].dense_shape

        decoded = tf.SparseTensorValue(indices = d_i, values = d_v, dense_shape = d_s)
        trans = convert2ListLabels(decoded)
        print(trans)
        #
        str_result = ""
        for item in trans:
            # str_result += meta.mapOrder2Char(item)
            seq = list(map(meta.mapOrder2Char, item))
            str_result = ''.join(seq)
            #
    return str_result
