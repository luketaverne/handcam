import tensorflow as tf

class LSTMModel():
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables sharing the parameters so that both graphs share the parameters.
    """

    def __init__(self, config, input_op, labels_op, seq_len_op, mode):
        """
        Basic setup.
        Args:
          config: Object containing configuration parameters.
        """
        assert mode in ["training", "validation", "inference"]
        self.config = config
        self.inputs = input_op
        self.labels = labels_op
        self.seq_lengths = seq_len_op
        self.mode = mode
        self.reuse = self.mode == "validation"

    def lstm_cell(self):
        with tf.variable_scope('rnn_cell', reuse=self.reuse, initializer=tf.contrib.layers.xavier_initializer()):
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.config['num_hidden_units'], reuse=self.reuse)

    def build_rnn_model(self):

        with tf.variable_scope('rnn_stack', reuse=self.reuse, initializer=tf.contrib.layers.xavier_initializer()):
            if self.config['num_layers'] > 1:
                rnn_cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.config['num_layers'])])
            else:
                rnn_cell = self.lstm_cell()
            self.model_rnn, self.rnn_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=self.inputs,
                dtype=tf.float32,
                sequence_length=self.seq_lengths,
                time_major=False,
                swap_memory=True)
            # initial_state = state = rnn_cell.zero_state(self.seq_lengths.shape[0], tf.float32)
            # Fetch output of the last step.
            if self.config['loss_type'] == 'last_step':
                self.rnn_prediction = tf.gather_nd(self.model_rnn,
                                                   tf.stack([tf.range(self.config['batch_size']), self.seq_lengths - 1],
                                                            axis=1))
            elif self.config['loss_type'] == 'average':
                self.rnn_prediction = self.model_rnn
            else:
                print("Invalid loss type")
                raise Exception

            print("rnn_prediction shape: " + str(self.rnn_prediction.shape))


    def build_model(self):
        self.build_rnn_model()
        # Calculate logits
        with tf.variable_scope('logits', reuse=self.reuse, initializer=tf.contrib.layers.xavier_initializer()):
            self.logits = tf.layers.dense(inputs=self.rnn_prediction, units=self.config['num_class_labels'],
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          bias_initializer=tf.contrib.layers.xavier_initializer())

        print("Logits shape: " + str(self.logits.shape))

    def loss(self):
        if self.mode is not "inference":
            # Loss calculations: cross-entropy
            if self.config['loss_type'] == 'last_step':
                with tf.name_scope("cross_entropy_loss"):
                    self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels[:, 0, :]))

            elif self.config['loss_type'] == 'average':

                with tf.name_scope("sequence_loss"):
                    self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                                 targets=tf.argmax(self.labels, 2),
                                                                 weights=tf.sequence_mask(lengths=self.seq_lengths,
                                                                                          maxlen=tf.reduce_max(self.seq_lengths),
                                                                                          dtype=tf.float32),
                                                                 # average_across_timesteps=False,
                                                                 # average_across_batch=False
                                                                 )

            print("loss shape: " + str(self.loss.shape))

                # Accuracy calculations.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            if self.config['loss_type'] == 'last_step':
                self.predictions = tf.argmax(self.logits, 1, name="predictions")
            elif self.config['loss_type'] == 'average':
                self.predictions = tf.argmax(self.logits, 2, name="predictions")
            print("predictions shape" + str(self.predictions.shape))

            # if self.mode is not "inference":
                # Return a bool tensor with shape [batch_size] that is true for the
                # correct predictions.
            if self.config['loss_type'] == 'last_step':
                self.correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels[:, 0, :], 1))
                print("correct predictions shape: " + str(self.correct_predictions.shape))
                # Number of correct predictions in order to calculate average accuracy afterwards.
                self.num_correct_predictions = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))
                # Calculate the accuracy per minibatch.
                self.batch_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
            elif self.config['loss_type'] == 'average':
                self.correct_predictions = tf.equal(tf.argmax(self.logits, 2), tf.argmax(self.labels, 2))
                print("correct predictions shape: " + str(self.correct_predictions.shape))
                # Number of correct predictions in order to calculate average accuracy afterwards.
                self.num_correct_predictions = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))
                # Calculate the accuracy per minibatch.
                self.batch_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

    def build_graph(self):
        self.build_model()
        self.loss()
        self.num_parameters()

    def num_parameters(self):
        self.num_parameters = 0
        # iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # mutiplying dimension values
            self.num_parameters += local_parameters

