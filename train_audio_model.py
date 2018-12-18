import argparse
from datetime import datetime
import json
import os
import sys
import time
import random
import audio_reader
from audio_model import *
import param
BATCH_SIZE = 2
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-2
WAVENET_PARAMS = param.wavenet_param_dir()
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False



def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
#    parser.add_argument('--histograms', type=_str_to_bool, default=False,
#                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='The number of training epoches is'
                             + str(100) + '.')
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def main():
    args = get_arguments()
    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)
#    audio_batches = audio_reader.read_audios(receptive_field= 5116,
#                sample_size = args.sample_size,batch_size= args.batch_size)
    audio_batches = np.load(param.get_audio_dir()+'/for_colab.npy')

    audio_batches_2 = []
    for batch in audio_batches:
        for i in batch:
            audio_batches_2.append(i)
    batch_num = len(audio_batches_2)// args.batch_size
    audio_batches = []
    for i in range(batch_num):
        audio_batches.append(np.array(audio_batches_2[i*args.batch_size:(i+1)*args.batch_size]))

    audio_batch = tf.placeholder(tf.float32,[args.batch_size,args.sample_size+5116,1])

    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"])

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    optimizer_factory = {'adam': create_adam_optimizer,
                         'sgd': create_sgd_optimizer,
                         'rmsprop': create_rmsprop_optimizer}
    loss = net.loss(input_batch=audio_batch,#shape = batch_size,length,1
                    l2_regularization_strength=args.l2_regularization_strength)
    lr = tf.Variable(args.learning_rate,trainable= False)
    optimizer = optimizer_factory[args.optimizer](
                    learning_rate=lr,
                    momentum=args.momentum)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(loss, var_list=trainable)

    save_dir = param.get_save_dir()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    saved_global_step = load(saver, sess, save_dir)
    if saved_global_step == None:
        saved_global_step = -1

    random.shuffle(audio_batches)
    audio_len = len(audio_batches)
    train_batches = audio_batches[0:int(audio_len*0.75)]
    test_batches = audio_batches[int(audio_len*0.75):]

    t1 = time.time()
    for epoch in range(saved_global_step + 1,args.n_epoch):
        train_loss = 0
        test_loss = 0
        random.shuffle(train_batches)
        if epoch  % 30 == 0 and epoch != 0:
            sess.run(lr.assign(sess.run(lr) * 0.1))
        for batch_id,batch in enumerate(train_batches):
            batch = np.array(batch)
            batch = np.reshape(batch,[args.batch_size,args.sample_size+5116,1])
            loss_value, _ = sess.run(
                [loss, optim],
            feed_dict={audio_batch:batch})
#            print('epoch = {},loss = {} batch = {}'.format(epoch,loss_value,batch_id))
            train_loss +=loss_value
        for batch_id, batch in enumerate(test_batches):
            batch = np.array(batch)
            batch = np.reshape(batch, [args.batch_size, 100000 + 5116, 1])
            loss_value = sess.run([loss],
                feed_dict={audio_batch: batch})
            test_loss += loss_value[0]
        t2 = time.time()
        print('epoch = {} train_loss = {} avg_loss = {} avg_test_loss = {} cost_time = {} lr = {}'.format(
            epoch,train_loss,train_loss/len(train_batches),test_loss/len(test_batches),t2 - t1,sess.run(lr)))
        t1 = time.time()
        if epoch % 10 == 0 and epoch != 0:
            save(saver, sess, save_dir, epoch)
            last_saved_epoch = epoch
            print('Model Trained and Saved')

if __name__ == '__main__':
    main()