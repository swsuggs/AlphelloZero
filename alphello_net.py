import tensorflow as tf
import tensorflow.contrib.slim as slim
from softmax import softmax

from data_manager import Data_Manager

class Othello_Network():

    def __init__(self, board_dim=8, time_steps=1, n_filters=256, conv_size=3, n_res=40, c=.1):
        """
        :param board_dim: dimension of game board
        :param time_steps: number of time steps kept in state history
        :param n_filters: number of convolutional filters per conv layer
        :param conv_size: size of convolutions
        :param n_res: number of residual layers
        :param c: regularization scale constant
        """
        self.board_dim = board_dim
        self.time_steps = time_steps
        self.losses = None
        self.n_conv_filters = n_filters
        self.conv_size = conv_size
        self.n_res_layers = n_res
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=c)
        self.dm = Data_Manager(max_size=(board_dim**2 - 4)*1000)  # moves per game TIMES num games to save

        # --------------
        # Make Network
        # --------------

        with tf.Graph().as_default() as net1_graph:
            self.input_layer = tf.placeholder(
                                         shape=[None, self.board_dim, self.board_dim, (self.time_steps * 2 + 1)],
                                         dtype=tf.float32, name='input')
            self.net = self._add_conv_layer(self.input_layer, name='conv1')
            for i in range(self.n_res_layers):
                self.net = self._add_res_layer(self.net, name='res{}'.format(i + 1))

            self.policy_logits = self._policy_head(self.net)
            self.value_estimate = self._value_head(self.net)

            self.mcts_pi = tf.placeholder(shape=[None, (self.board_dim**2 + 1)], dtype=tf.float32, name='pi')
            self.winner_z = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='z')

            # Loss, composed of cross entropy, mse, and regularization
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.mcts_pi, logits=self.policy_logits)
            mse = tf.losses.mean_squared_error(self.winner_z, self.value_estimate)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = tf.reduce_mean(mse - xent + sum(reg_losses))

            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)  # tune learning rate

            # more ops
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        # initialize session
        self.sess = tf.Session(graph=net1_graph)
        self.sess.run(self.init_op)


    def _add_conv_layer(self, input_layer, name=None):
        """
        Create a general convolutional layer.

        :param input_layer: the previous network layer to build on
        :param name: name of this layer
        :return: the output layer
        """
        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=self.n_conv_filters,
            kernel_size=[self.conv_size, self.conv_size],
            padding="same",
            kernel_regularizer=self.regularizer,
            name=name)

        bn = slim.batch_norm(conv)
        output = tf.nn.relu(bn)

        return output

    def _add_res_layer(self, input_layer, name):
        """
        Create a general residual layer

        :param input_layer: the previous network layer to build on
        :param name: the name of this layer
        :return: the output layer
        """

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=self.n_conv_filters,
            kernel_size=[self.conv_size, self.conv_size],
            padding="same",
            kernel_regularizer=self.regularizer,
            name='{}_c1'.format(name))

        bn1 = slim.batch_norm(conv1)
        relu1 = tf.nn.relu(bn1)

        conv2 = tf.layers.conv2d(
            inputs=relu1,
            filters=self.n_conv_filters,
            kernel_size=[self.conv_size, self.conv_size],
            padding="same",
            kernel_regularizer=self.regularizer,
            name='{}_c2'.format(name))

        bn2 = slim.batch_norm(conv2)
        skip_connection = input_layer + bn2
        output = tf.nn.relu(skip_connection)

        return output

    def _policy_head(self, input_layer):
        """
        Estimate move probability distribution by applying the policy head network to input_layer.

        :param input_layer: layer to apply policy head to.
        :param softmax: whether or not to softmax the logits into a probability distribution.
        :return: vector of board_dim * board_dim + 1 logits.
        """

        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=2,
            kernel_size=[1, 1],
            padding="same",
            kernel_regularizer=self.regularizer)

        bn = slim.batch_norm(conv)
        relu = tf.nn.relu(bn)

        fc = tf.layers.dense(inputs=tf.contrib.layers.flatten(relu), units=(self.board_dim * self.board_dim + 1),
                             kernel_regularizer=self.regularizer)

        return fc

    def _value_head(self, input_layer):
        """
        Estimate value of board state by applying value head network to input_layer.

        :param input_layer: the layer to apply value head to
        :return: scalar estimating value of board position (between -1 and 1)
        """

        conv = tf.layers.conv2d(
            inputs=input_layer,
            filters=1,
            kernel_size=[1, 1],
            padding="same",
            kernel_regularizer=self.regularizer)

        bn = slim.batch_norm(conv)
        relu1 = tf.nn.relu(bn)

        fc1 = tf.layers.dense(inputs=tf.contrib.layers.flatten(relu1), units=256, kernel_regularizer=self.regularizer)
        relu2 = tf.nn.relu(fc1)
        fc2 = tf.layers.dense(inputs=relu2, units=1, kernel_regularizer=self.regularizer)

        output = tf.nn.tanh(fc2)
        return output

    def add_training_data(self, states, pis, zs):
        """
        Add data to the data manager.
        :param states: N X board_size X board_size * 3 array of states
        :param pis: N X (board_size**2 + 1) array of move distributions
        :param zs: N X 1 array of winners.
        """
        self.dm.add_data(states, pis, zs)

    def estimate_policy(self, state, soft=True):
        """
        Estimate policy distribution for a state.
        :param state: Must be batch_size X board_size X board_size X 3, even if batch size is 1
        :param soft: Whether to softmax the logits before returning or not.  Default True.
        :return: estimated policy distribution
        """

        feed_dict = {self.input_layer: state}
        logits = self.sess.run([self.policy_logits], feed_dict=feed_dict)[0]
        if soft:
            policy = softmax(logits)
            return policy
        else: return logits


    def estimate_value(self, state):
        """
        Estimate value of a state (probability of current player winning)
        :param state: Must be batch_size X board_size X board_size X 3, even if batch size is 1
        :return: estimated value, betwen -1 and 1
        """

        feed_dict = {self.input_layer: state}
        return self.sess.run([self.value_estimate], feed_dict=feed_dict)[0]


    def save_weights(self, path="/tmp/model.ckpt"):
        """
        Save the current weights of the network
        :param path: the path to saved files
        """
        self.saver.save(self.sess, path)


    def load_weights(self, path="/tmp/model.ckpt"):
        """
        Load network weights from file.
        :param path: the path to saved files
        """
        self.saver.restore(self.sess, path)


    def train(self, n_iters=1000, batch_size=1024, verbose=True):
        """
        Train the network some amount.
        :param n_iters: How many batches to train on.
        :param batch_size: Size of each batch.
        :return: list of losses
        """

        losses = []

        # sample mini-batch of 2048
        for i in range(n_iters):

            state_batch, pi_batch, z_batch = self.dm.get_batch(batch_size)
            feed_dict = {self.input_layer: state_batch, self.mcts_pi: pi_batch, self.winner_z: z_batch}
            l, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)  # probably don't need to run loss every time
            losses.append(l)

            if verbose:
                if i % 100 == 0:
                    print("{}: loss: {}".format(i, l))

        if self.losses is not None:
            self.losses.extend(losses)
        else:
            self.losses = losses
