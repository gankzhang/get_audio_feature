import numpy as np
import tensorflow as tf


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32):
        '''Initializes the WaveNet model.

                Args:
                    batch_size: How many audio files are supplied per batch
                        (recommended: 1).
                    dilations: A list with the dilation factor for each layer.
                    filter_width: The samples that are included in each convolution,
                        after dilating.
                    residual_channels: How many filters to learn for the residual.
                    dilation_channels: How many filters to learn for the dilated
                        convolution.
                    skip_channels: How many filters to learn that contribute to the
                        quantized softmax output.
                    quantization_channels: How many amplitude values to use for audio
                        quantization and the corresponding one-hot encoding.
                        Default: 256 (8-bit quantization).
                    use_biases: Whether to add a bias layer to each convolution.
                        Default: False.
                    scalar_input: Whether to use the quantized waveform directly as
                        input to the network instead of one-hot encoding it.
                        Default: False.
                    initial_filter_width: The width of the initial filter of the
                        convolution applied to the scalar input. This is only relevant
                        if scalar_input=True.
                    histograms: Whether to store histograms in the summary.
                        Default: False.
                    global_condition_channels: Number of channels in (embedding
                        size) of global conditioning vector. None indicates there is
                        no global conditioning.
                    global_condition_cardinality: Number of mutually exclusive
                        categories to be embedded in global condition embedding. If
                        not None, then this implies that global_condition tensor
                        specifies an integer selecting which of the N global condition
                        categories, where N = global_condition_cardinality. If None,
                        then the global_condition tensor is regarded as a vector which
                        must have dimension global_condition_channels.

                '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width

        self.receptive_field = self.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.variables = self._create_variables()

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):

            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_channels = self.quantization_channels
                    initial_filter_width = self.filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,#2
                             self.residual_channels,#32
                             self.dilation_channels])#32
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.quantization_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.quantization_channels])
                var['postprocessing'] = current

        return var




    def calculate_receptive_field(self,filter_width, dilations, scalar_input,#???????????????????????????????????????????????????????
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field
    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.
        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)
    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             global_conditioning_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']#filter 和 gate 核维度是一样的,但是后续一个用了sigmoid，一个用了tanh
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)


        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)#点乘

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias


        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed


    def _create_network(self, input_batch):
        '''Construct the WaveNet network.'''
        self.outputs = []
        current_layer = input_batch
        #这里没有用下面的output，因为很多选项并不考虑，所以代码简化了。
        # Pre-process the input with a regular convolution
        if self.scalar_input:
            initial_channels = 1
        else:
            initial_channels = self.quantization_channels

        current_layer = self._create_causal_layer(current_layer)#一般的卷积层

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        # output 为skip output， current layer 为 dense output
        with tf.name_scope('dilated_stack'):#core 最核心的扩大卷积
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                         output_width)
                    self.outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']


            total = sum(self.outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)
        return conv2


    def loss(self,
             input_batch,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            encoded_input = mu_law_encode(input_batch,
                                          self.quantization_channels)#某种类似sigmoid归于化的变换
            encoded = self._one_hot(encoded_input)
            if self.scalar_input:#default is False
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])

            raw_output = self._create_network(network_input)


            # Cut off the samples corresponding to the receptive field
            # for the first predicted sample.
            target_output = tf.slice(
                tf.reshape(
                    encoded,
                    [self.batch_size, -1, self.quantization_channels]),
                [0, self.receptive_field, 0],
                [-1, -1, -1])
            target_output = tf.reshape(target_output,
                                       [-1, self.quantization_channels])
            prediction = tf.reshape(raw_output,
                                    [-1, self.quantization_channels])
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction,
                labels=target_output)
            reduced_loss = tf.reduce_mean(loss)

            tf.summary.scalar('loss', reduced_loss)

            if l2_regularization_strength is None:
                return reduced_loss
            else:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not('bias' in v.name)])

                # Add the regularization term to the loss
                total_loss = (reduced_loss +
                              l2_regularization_strength * l2_loss)

                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('total_loss', total_loss)

                return total_loss


    def output(self,input_batch,name='wavenet'):
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            encoded_input = mu_law_encode(input_batch,
                                          self.quantization_channels)  # 某种类似sigmoid归于化的变换
            encoded = self._one_hot(encoded_input)
            if self.scalar_input:  # default is False
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded

            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])

            raw_output = self._create_network(network_input)
            return raw_output

    '''
    def output(self,input_batch,name='wavenet'):
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            encoded_input = mu_law_encode(input_batch,
                                          self.quantization_channels)#某种类似sigmoid归于化的变换
            encoded = self._one_hot(encoded_input)
            if self.scalar_input:#default is False
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded
            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0],
                                     [-1, network_input_width, -1])

            outputs = []
            current_layer = network_input

            # Pre-process the input with a regular convolution
            if self.scalar_input:
                initial_channels = 1
            else:
                initial_channels = self.quantization_channels

            current_layer = self._create_causal_layer(current_layer)  # 一般的卷积层
            output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

            # Add all defined dilation layers.
            # output 为skip output， current layer 为 dense output
            with tf.name_scope('dilated_stack'):  # core 最核心的扩大卷积
                for layer_index, dilation in enumerate(self.dilations):
                    with tf.name_scope('layer{}'.format(layer_index)):
                        output, current_layer = self._create_dilation_layer(
                            current_layer, layer_index, dilation,
                            output_width)
                        outputs.append(output)
            return outputs
'''

def causal_conv(value, filter_, dilation):
    filter_width = tf.shape(filter_)[0]
    if dilation > 1:
        transformed = time_to_batch(value, dilation)
        conv = tf.nn.conv1d(transformed, filter_, stride=1,
                            padding='VALID')#VALID 为 不添加
        restored = batch_to_time(conv, dilation)
    else:
        restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
    # Remove excess elements at the end.
    out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
    result = tf.slice(restored,
                      [0, 0, 0],
                      [-1, out_width, -1])
    return result

def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable




def time_to_batch(value, dilation, name=None):
    shape = tf.shape(value)
    pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
    padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])#修改之后，shape[1] % dilation == 0
    reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
    transposed = tf.transpose(reshaped, perm=[1, 0, 2])
    return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):#和上面的time_to_batch互为逆运算
    shape = tf.shape(value)
    prepared = tf.reshape(value, [dilation, -1, shape[2]])
    transposed = tf.transpose(prepared, perm=[1, 0, 2])
    return tf.reshape(transposed,
                      [tf.div(shape[0], dilation), -1, shape[2]])

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = tf.to_float(quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
    signal = tf.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return tf.to_int32((signal + 1) / 2 * mu + 0.5)

def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}