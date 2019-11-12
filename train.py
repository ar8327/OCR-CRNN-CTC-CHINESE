from core.model import *
import core.meta as meta
import os
from core.loadData import *
import time
import random
import core.util
from tensorflow.python.framework import graph_util

T_DEBUG = False

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# model save-path
if not os.path.exists(meta.model_recog_dir): os.mkdir(meta.model_recog_dir)
#
# graph
graph,y,train_op = ModelTrain()
if T_DEBUG:
    data_train = load_data('/home/xsooy/newOCREXP/tools/data_rects_valid/')
else:
    data_train = load_data('/home/xsooy/newOCREXP/tools/data_rects_train/')
# data_valid = load_data('PATH')
#
# restore and train
with graph.as_default():

    var_list = tf.trainable_variables()
    # for g in var_list:
    #     print(g.name)
    # print("===============================")
    g_list = tf.global_variables()
    # for g in g_list:
    #     print(g.name)
    bn_moving_vars = [g for g in g_list if 'mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'variance' in g.name]
    print("bn_moving_vars",bn_moving_vars)
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list,max_to_keep=10)


    with tf.Session() as sess:

        # print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])

        global_step = tf.train.get_or_create_global_step()
        learning_rate = graph.get_tensor_by_name('learning_rate:0')
        x = graph.get_tensor_by_name('x-input:0')
        w = graph.get_tensor_by_name('w-input:0')
        # y = graph.get_tensor_by_name('y-input:0')
        # train_op = graph.get_tensor_by_name('adam:0')
        loss = graph.get_tensor_by_name('loss:0')
        #
        tf.global_variables_initializer().run()
        sess.run(tf.assign(learning_rate, tf.constant(meta.LEARNING_RATE_BASE, dtype=tf.float32)))
        #
        # restore with saved data
        ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
        #
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        #
        #
        print('begin to train ...')
        #
        # variables
        #
        # y_s = self.graph.get_tensor_by_name('y-input/shape:0')
        # y_i = self.graph.get_tensor_by_name('y-input/indices:0')
        # y_v = self.graph.get_tensor_by_name('y-input/values:0')
        #
        # <tf.Operation 'y-input/shape' type=Placeholder>,
        # <tf.Operation 'y-input/values' type=Placeholder>,
        # <tf.Operation 'y-input/indices' type=Placeholder>]
        #
        #
        num_samples = len(data_train['x'])
        #
        # start training
        start_time = time.time()
        begin_time = start_time
        #
        step = sess.run(global_step)
        #
        train_step_half = int(meta.TRAINING_STEPS * 0.5)
        train_step_quar = int(meta.TRAINING_STEPS * 0.75)
        #
        while step <= meta.TRAINING_STEPS:
            #
            if step == train_step_half:
                sess.run(tf.assign(learning_rate, tf.constant(meta.LEARNING_RATE_BASE / 10, dtype=tf.float32)))
            if step == train_step_quar:
                sess.run(
                    tf.assign(learning_rate, tf.constant(meta.LEARNING_RATE_BASE / 100, dtype=tf.float32)))
            #
            # save and validate

            if step % save_freq == 0:
                print('save model to ckpt ...')
                saver.save(sess, os.path.join(meta.model_recog_dir, meta.model_recog_name), \
                           global_step=step)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names = \
                                                                           ['rnn_logits/BiasAdd','seq_len',\
                                                                            'CTCBeamSearchDecoder'])
                with tf.gfile.FastGFile(core.meta.model_recog_pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())



            if step % valid_freq == 0 and step != 0:
                samples = os.listdir('/home/xsooy/newOCREXP/tools/data_rects_valid/images/')
                s = random.choice(list(samples))
                _y = s.split('_')[1]
                print("predict = ",core.util.predict(graph=ModelPredict(), image_in='/home/xsooy/newOCREXP/tools/data_rects_valid/images/'+s))
                print("y = ",_y)


                pass
                #
                # print('save model to ckpt ...')
                # saver.save(sess, os.path.join(meta.model_recog_dir, meta.model_recog_name), \
                #            global_step=step)
                #

            # train
            index_batch = random.sample(range(num_samples), meta.BATCH_SIZE)

            images = [data_train['x'][i] for i in index_batch]
            targets = [data_train['y'][i] for i in index_batch]

            w_arr = [item.shape[1] for item in images]
            max_w = max(w_arr)
            img_padd = []
            for item in images:
                if item.shape[1] != max_w:
                    img_zeros = np.zeros(shape=[meta.height_norm, max_w - item.shape[1], 3], dtype=np.float32)
                    item = np.concatenate([item, img_zeros], axis=1)
                img_padd.append(item)
            images = img_padd

            tsv = core.util.convert2SparseTensorValue(targets)
            feed_dict = {x: images, w: w_arr, y: tsv}

            # sess.run
            _, loss_value, step, lr = sess.run([train_op, loss, global_step, learning_rate], \
                                               feed_dict)
            #
            if step % meta.loss_freq == 0: #Show train info
                #
                curr_time = time.time()
                #
                print('step: %d, loss: %g, lr: %g, sect_time: %.1f, total_time: %.1f' %
                      (step, loss_value, lr, curr_time - begin_time, curr_time - start_time))
                #
                begin_time = curr_time
                #
            #
