import tensorflow as tf

class VGG():
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
        """
        Builds a model by using tf.layers API.

        Note that in mnist_fc_with_summaries.ipynb weights and biases are
        defined manually. tf.layers API follows similar steps in the background.
        (you can check the difference between tf.nn.conv2d and tf.layers.conv2d)
        """
        with tf.variable_scope("cnn_model", reuse=self.reuse, initializer=self.config['initializer']):
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=self.config['vgg_filters'][0],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=self.config['vgg_filters'][1],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="same")

            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=self.config['vgg_filters'][2],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=self.config['vgg_filters'][3],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding="same")

            conv5 = tf.layers.conv2d(
                inputs=pool3,
                filters=self.config['vgg_filters'][4],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            conv6 = tf.layers.conv2d(
                inputs=conv5,
                filters=self.config['vgg_filters'][5],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

            conv7 = tf.layers.conv2d(
                inputs=pool4,
                filters=self.config['vgg_filters'][6],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            conv8 = tf.layers.conv2d(
                inputs=conv7,
                filters=self.config['vgg_filters'][7],
                kernel_size=3,
                padding="same",
                activation=tf.nn.relu)

            # pool5 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2)

            pool5_flat = tf.reshape(conv8, [-1, 5 * 5 * self.config['vgg_filters'][7]])

            dense1 = tf.layers.dense(inputs=pool5_flat, units=self.config['num_hidden_units'], activation=tf.nn.relu)

            dropout1 = tf.layers.dropout(inputs=dense1, rate=self.config['dropout_rate'], training=self.is_training)

            dense2 = tf.layers.dense(inputs=dropout1, units=self.config['num_hidden_units'], activation=tf.nn.relu)

            dropout2 = tf.layers.dropout(inputs=dense2, rate=self.config['dropout_rate'], training=self.is_training)

            logits = tf.layers.dense(inputs=dense2, units=20)

            return dropout2

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