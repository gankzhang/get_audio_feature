import numpy as np
import tensorflow as tf
import param
from audio_model import *
import random
import time

def main():
    audio_batches = np.load(param.get_audio_dir()+'/for_colab.npy')
    audio_batches_2 = []
    for batch in audio_batches:
        for i in batch:
            audio_batches_2.append(i)
    batch_num = len(audio_batches_2)
    audio_batches = []
    for i in range(batch_num):
        audio_batches.append(np.array(audio_batches_2[i]))



    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    filter_width = 32
    quantization_channels = 16
    residual_channels = 16
    skip_channels = 16

    audio_batch = tf.placeholder(tf.float32, [1, 105116, 1])
    encoded_input = mu_law_encode(audio_batch,quantization_channels)
    encoded = tf.one_hot(
        encoded_input,depth=quantization_channels,dtype=tf.float32)
    encoded = tf.reshape(encoded, [1, -1, quantization_channels])
    network_input_width = tf.shape(encoded)[1] - 1
    network_input = tf.slice(encoded, [0, 0, 0],[-1, network_input_width, -1])
    filter_1 = tf.Variable(initializer(shape=[filter_width,quantization_channels,residual_channels]))
    filter_2 = tf.Variable(initializer(shape=[filter_width,quantization_channels,residual_channels]))
    current_layer = network_input
    current_layer = causal_conv(current_layer,filter_1,1)#dilation = 4
    current_layer = causal_conv(current_layer, filter_2, 1)  # dilation = 4

    w1 = tf.Variable(initializer(shape=[1, skip_channels, skip_channels]))
    w2 = tf.Variable(initializer(shape=[1, skip_channels, skip_channels]))
    b1 = tf.Variable(initializer(shape=[skip_channels]))
    b2 = tf.Variable(initializer(shape=[skip_channels]))

    transformed1 = tf.nn.relu(current_layer)
    conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
    conv1 = tf.add(conv1, b1)
    transformed2 = tf.nn.relu(conv1)
    conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
    raw_output = tf.add(conv2, b2)

    prediction = tf.reshape(raw_output,[-1, quantization_channels])

    target_output = tf.slice(
        tf.reshape(encoded,[1, -1, quantization_channels]),
        [0, (filter_width - 1)* 2 + 1, 0],[-1, -1, -1])
    target_output = tf.reshape(target_output,[-1, quantization_channels])
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction,
        labels=target_output)
    reduced_loss = tf.reduce_mean(loss)
    lr = tf.Variable(0.01, trainable=False)
    optimizer_factory = {'adam': create_adam_optimizer,
                         'sgd': create_sgd_optimizer,
                         'rmsprop': create_rmsprop_optimizer}
    optimizer = optimizer_factory['adam'](
        learning_rate= lr,
        momentum=0.9)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(reduced_loss, var_list=trainable)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)
    random.shuffle(audio_batches)
    train_batches = audio_batches[0:60]
    test_batches = audio_batches[60:]
    t1 = time.time()
    for epoch in range(400):
        train_loss = 0
        test_loss = 0
        if epoch  % 100 == 0 and epoch != 0:
            sess.run(lr.assign(sess.run(lr) * 0.1))
        random.shuffle(train_batches)
        for batch_id, batch in enumerate(train_batches):
            batch = np.array(batch)
            batch = np.reshape(batch, [1, 100000 + 5116, 1])
            loss_value, _ = sess.run(
                [reduced_loss, optim],
                feed_dict={audio_batch: batch})
#            print('epoch = {},loss = {} batch = {}'.format(epoch,loss_value,batch_id))
            train_loss += loss_value
        for batch_id, batch in enumerate(test_batches):
            batch = np.array(batch)
            batch = np.reshape(batch, [1, 100000 + 5116, 1])
            loss_value = sess.run([reduced_loss],
                feed_dict={audio_batch: batch})
            # print('epoch = {},loss = {} batch = {}'.format(epoch,loss_value,batch_id))
            test_loss += loss_value[0]

        t2 = time.time()
        print('epoch = {} train_loss = {} avg_loss = {} avg_test_loss = {} cost_time = {}'.format(epoch, train_loss,
                                                                               train_loss / len(train_batches),
                                                                               test_loss / len(test_batches),
                                                                               t2 - t1))
        t1 = time.time()

if __name__ == '__main__':
    main()