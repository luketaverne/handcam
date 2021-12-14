import tensorflow as tf

class CNN_depth():
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables sharing the parameters so that both graphs share the parameters.
    """

    def __init__(self, config, input_op, mode):
        """
        Basic setup.
        Args:
          config: Object containing configuration parameters.
        """
        assert mode in ["training", "validation", "inference"]
        self.config = config
        self.inputs = input_op
        self.mode = mode
        self.is_training = self.mode == "training"
        self.reuse = self.mode == "validation"

    def build_model(self, input_layer):
        rgb = input_layer[...,0:3]
        depth = input_layer[...,3:]
        with tf.variable_scope("cnn_model", reuse=self.reuse, initializer=self.config['initializer']):
            # Convolutional Layer #1
            # Computes 32 features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 80, 80, num_channels]
            # Output Tensor Shape: [batch_size, 40, 40, num_filter1]
            with tf.variable_scope("rgb", reuse=self.reuse, initializer=self.config['initializer']):
                print(rgb.shape)
                with tf.variable_scope('bn_rgb'):
                    # phase_train = tf.placeholder(tf.bool, name='phase_train')

                    normed_rgb = tf.contrib.layers.batch_norm(rgb, is_training=self.is_training, fused=True, updates_collections=None)
                conv1 = tf.layers.conv2d(
                    inputs=normed_rgb,
                    filters=self.config['cnn_filters'][0],
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

                # Input Tensor Shape: [batch_size, 40, 40, num_filter1]
                # Output Tensor Shape: [batch_size, 20, 20, num_filter2]
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=self.config['cnn_filters'][1],
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')

                # Input Tensor Shape: [batch_size, 20, 20, num_filter2]
                # Output Tensor Shape: [batch_size, 10, 10, num_filter3]
                conv3 = tf.layers.conv2d(
                    inputs=pool2,
                    filters=self.config['cnn_filters'][2],
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)

                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='same')

                # Input Tensor Shape: [batch_size, 10, 10, num_filter3]
                # Output Tensor Shape: [batch_size, 5, 5, num_filter4]
                conv4 = tf.layers.conv2d(
                    inputs=pool3,
                    filters=self.config['cnn_filters'][3],
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)

                pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='same')
            with tf.variable_scope("depth", reuse=self.reuse, initializer=self.config['initializer']):
                print(depth.shape)
                with tf.variable_scope('bn_depth'):
                    # phase_train = tf.placeholder(tf.bool, name='phase_train')

                    normed_depth = tf.contrib.layers.batch_norm(depth, is_training=self.is_training, fused=True, updates_collections=None)
                conv1_d = tf.layers.conv2d(
                    inputs=normed_depth,
                    filters=self.config['cnn_filters'][0],
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)

                pool1_d = tf.layers.max_pooling2d(inputs=conv1_d, pool_size=[2, 2], strides=2, padding='same')

                # Input Tensor Shape: [batch_size, 40, 40, num_filter1]
                # Output Tensor Shape: [batch_size, 20, 20, num_filter2]
                conv2_d = tf.layers.conv2d(
                    inputs=pool1_d,
                    filters=self.config['cnn_filters'][1],
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)

                pool2_d = tf.layers.max_pooling2d(inputs=conv2_d, pool_size=[2, 2], strides=2, padding='same')

                # Input Tensor Shape: [batch_size, 20, 20, num_filter2]
                # Output Tensor Shape: [batch_size, 10, 10, num_filter3]
                conv3_d = tf.layers.conv2d(
                    inputs=pool2_d,
                    filters=self.config['cnn_filters'][2],
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)

                pool3_d = tf.layers.max_pooling2d(inputs=conv3_d, pool_size=[2, 2], strides=2, padding='same')

                # Input Tensor Shape: [batch_size, 10, 10, num_filter3]
                # Output Tensor Shape: [batch_size, 5, 5, num_filter4]
                conv4_d = tf.layers.conv2d(
                    inputs=pool3_d,
                    filters=self.config['cnn_filters'][3],
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)

                pool4_d = tf.layers.max_pooling2d(inputs=conv4_d, pool_size=[2, 2], strides=2, padding='same')

            # Flatten tensor into a batch of vectors
            # Input Tensor Shape: [batch_size, 5, 5, num_filter4]
            # Output Tensor Shape: [batch_size, 5 * 5 * num_filter4]

            conv_flat_rgb = tf.reshape(pool4, [-1, 7 * 7 * self.config['cnn_filters'][3]])
            conv_flat_depth = tf.reshape(pool4_d, [-1, 7 * 7 * self.config['cnn_filters'][3]])
            conv_flat = tf.concat([conv_flat_rgb, conv_flat_depth], axis=1)
            with tf.variable_scope('bn_flat'):
                # phase_train = tf.placeholder(tf.bool, name='phase_train')

                normed_flat = tf.contrib.layers.batch_norm(conv_flat, is_training=self.is_training, fused=True, updates_collections=None)
            # Add dropout operation;
            # dropout = tf.layers.dropout(inputs=normed_flat, rate=self.config['dropout_rate'], training=self.is_training)

            # Dense Layer
            # Densely connected layer with <num_hidden_units> neurons
            # Input Tensor Shape: [batch_size, 5 * 5 * num_filter4]
            # Output Tensor Shape: [batch_size, num_hidden_units]
            dense = tf.layers.dense(inputs=normed_flat, units=self.config['num_hidden_units'],
                                    activation=tf.nn.relu)

            self.cnn_model = dense
            return dense

    def build_graph(self):
        """
        CNNs accept inputs of shape (batch_size, height, width, num_channels). However, we have inputs of shape
        (batch_size, sequence_length, height, width, num_channels) where sequence_length is inferred at run time.
        We need to iterate in order to get CNN representations. Similar to python's map function, "tf.map_fn"
        applies a given function on each entry in the input list.
        """
        # For the first time create a dummy graph and then share the parameters everytime.
        if self.is_training:
            self.reuse = False
            self.build_model(self.inputs[0])
            self.reuse = True

        # CNN takes a clip as if it is a batch of samples.
        # Have a look at tf.map_fn (https://www.tensorflow.org/api_docs/python/tf/map_fn)
        # You can set parallel_iterations or swap_memory in order to make it faster.
        # Note that back_prop argument is True in order to enable training of CNN.
        self.cnn_representations = tf.map_fn(lambda x: self.build_model(x),
                                             elems=self.inputs,
                                             dtype=tf.float32,
                                             back_prop=True,
                                             swap_memory=True,
                                             parallel_iterations=2)

        return self.cnn_representations


class CNN_rgb():
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables sharing the parameters so that both graphs share the parameters.
    """

    def __init__(self, config, input_op, mode):
        """
        Basic setup.
        Args:
          config: Object containing configuration parameters.
        """
        assert mode in ["training", "validation", "inference"]
        self.config = config
        self.inputs = input_op
        self.mode = mode
        self.is_training = self.mode == "training"
        self.reuse = self.mode == "validation"

    def build_model(self, input_layer):
        with tf.variable_scope("cnn_model", reuse=self.reuse, initializer=self.config['initializer']):
            # Convolutional Layer #1
            # Computes 32 features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 80, 80, num_channels]
            # Output Tensor Shape: [batch_size, 40, 40, num_filter1]
            # with tf.variable_scope('bn_rgb'):
                # phase_train = tf.placeholder(tf.bool, name='phase_train')
            with tf.variable_scope('bn_rgb'):
                # normed_rgb = tf.layers.batch_normalization(input_layer, training=self.is_training, fused=True)
                normed_rgb = tf.contrib.layers.batch_norm(input_layer, is_training=self.is_training, fused=True, updates_collections=None)
            conv1 = tf.layers.conv2d(
                inputs=normed_rgb,
                filters=self.config['cnn_filters'][0],
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

            # Input Tensor Shape: [batch_size, 40, 40, num_filter1]
            # Output Tensor Shape: [batch_size, 20, 20, num_filter2]
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=self.config['cnn_filters'][1],
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')

            # Input Tensor Shape: [batch_size, 20, 20, num_filter2]
            # Output Tensor Shape: [batch_size, 10, 10, num_filter3]
            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=self.config['cnn_filters'][2],
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='same')

            # Input Tensor Shape: [batch_size, 10, 10, num_filter3]
            # Output Tensor Shape: [batch_size, 5, 5, num_filter4]
            conv4 = tf.layers.conv2d(
                inputs=pool3,
                filters=self.config['cnn_filters'][3],
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='same')

            conv_flat = tf.reshape(pool4, [-1, 7 * 7 * self.config['cnn_filters'][3]])
            with tf.variable_scope('bn_flat'):
                # phase_train = tf.placeholder(tf.bool, name='phase_train')

                # normed_flat = tf.layers.batch_normalization(conv_flat, training=self.is_training, fused=True, updates_collections=None)
                normed_flat = tf.contrib.layers.batch_norm(conv_flat, is_training=self.is_training, fused=True, updates_collections=None)
            # Add dropout operation;
            dropout = tf.layers.dropout(inputs=normed_flat, rate=self.config['dropout_rate'], training=self.is_training)

            # Dense Layer
            # Densely connected layer with <num_hidden_units> neurons
            # Input Tensor Shape: [batch_size, 5 * 5 * num_filter4]
            # Output Tensor Shape: [batch_size, num_hidden_units]
            dense = tf.layers.dense(inputs=dropout, units=self.config['num_hidden_units'],
                                    activation=tf.nn.relu)

            self.cnn_model = dense
            return dense

    def build_graph(self):
        """
        CNNs accept inputs of shape (batch_size, height, width, num_channels). However, we have inputs of shape
        (batch_size, sequence_length, height, width, num_channels) where sequence_length is inferred at run time.
        We need to iterate in order to get CNN representations. Similar to python's map function, "tf.map_fn"
        applies a given function on each entry in the input list.
        """
        # For the first time create a dummy graph and then share the parameters everytime.
        if self.is_training:
            self.reuse = False
            self.build_model(self.inputs[0])
            self.reuse = True

        # CNN takes a clip as if it is a batch of samples.
        # Have a look at tf.map_fn (https://www.tensorflow.org/api_docs/python/tf/map_fn)
        # You can set parallel_iterations or swap_memory in order to make it faster.
        # Note that back_prop argument is True in order to enable training of CNN.
        self.cnn_representations = tf.map_fn(lambda x: self.build_model(x),
                                             elems=self.inputs,
                                             dtype=tf.float32,
                                             back_prop=True,
                                             swap_memory=True,
                                             parallel_iterations=2)

        return self.cnn_representations
