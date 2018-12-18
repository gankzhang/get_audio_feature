from audio_reader import *
from train_audio_model import *
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')

    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')

    return parser.parse_args()

def get_feature(audio_address):
    args = get_arguments()
 #   sample_length = 1027
    sample_length = 1026
    sample_num = 32

    audio_batch = tf.placeholder(tf.float32, [1, sample_length, 1])
    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        scalar_input=wavenet_params["scalar_input"],
        initial_filter_width=wavenet_params["initial_filter_width"])

    outputs = net.output(input_batch=audio_batch)

    save_dir = param.get_save_dir()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    saved_global_step = load(saver, sess, save_dir)
    args = get_arguments()
    audios = read_audios(audio_dir=audio_address.split('\\')[0] + '\\' + audio_address.split('\\')[1], sample_size='all')
    audios = np.array(audios)
    audio = np.reshape(audios, [-1])
    length = audio.shape[0]

    step = (length - sample_length) // (sample_num + 1)
    inputs = []
#    for i in range(sample_num):
#        inputs.append(audio[step * (i + 1):step * (i + 1) + sample_length])
    inputs.append(audio[100000 - sample_length : 100000])
    # sample出多个符合长度的段

    # input -> output
    output_values = []
    for i in range(sample_num):
        audio_sample = np.reshape(inputs[i], [-1, sample_length, 1])
        output_value = sess.run([outputs], feed_dict={audio_batch: audio_sample})[0][0, -1, :]
        output_values.append(output_value)
    output_values = np.reshape(output_values, [-1])
    return output_values


def gen_feature(audio_address):
    files = find_files(audio_address)
    for file in files:
        output_values = get_feature(file)
        np.save(file[:-4]+'.npy',output_values)
        os.remove(file)
        print(file)


if __name__ == '__main__':
    gen_feature('./纯音乐的殿堂_900_918')





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
