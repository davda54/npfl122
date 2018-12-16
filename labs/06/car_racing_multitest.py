#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import collections
import random

import car_racing_evaluator


def rgb2gray(rgb, shape):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).reshape(shape[0], shape[1], shape[2])

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto(inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads)
        config.gpu_options.allow_growth = True

        self.session = tf.Session(graph=graph, config=config)

    def construct(self, args, convolution, hidden_size, state_shape, num_actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.prev_states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None])
            self.returns = tf.placeholder(tf.float32, [None])
            self.weights = tf.placeholder(tf.float32, [None])

            input = tf.concat([tf.image.resize_images(self.states, [32, 32]), tf.image.resize_images(self.prev_states, [32, 32])], axis=3)

            output = input
            for filters, kernel, stride in convolution:
                if filters == 0:
                    output = tf.layers.max_pooling2d(inputs=output, pool_size=[kernel, kernel], strides=stride)
                else:
                    output = tf.layers.conv2d(inputs=output, filters=filters, kernel_size=[kernel, kernel], strides=stride, padding=args.padding)
                    output = tf.nn.relu(output)

            output = tf.layers.flatten(output)
            output = tf.layers.dense(output, hidden_size, activation=tf.nn.relu)
            self.predicted_values = tf.layers.dense(output, num_actions, activation=None)

            self.softmax = tf.nn.softmax(self.predicted_values)

            # v_dense = tf.layers.dense(output, hidden_size, activation=tf.nn.relu)
            # a_dense = tf.layers.dense(output, hidden_size, activation=tf.nn.relu)
            #
            # v = tf.layers.dense(v_dense, 1, activation=None)
            # a = tf.layers.dense(a_dense, num_actions, activation=None)
            #
            # self.predicted_values = v + a - tf.reduce_mean(a, 1, keep_dims=True)

            loss = tf.losses.mean_squared_error(self.returns, tf.boolean_mask(self.predicted_values, tf.one_hot(self.actions, num_actions)), weights=self.weights)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def copy_variables_from(self, other):
        for variable, other_variable in zip(self.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                                            other.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
            variable.load(other_variable.eval(other.session), self.session)

    def predict(self, prev_states, states):
        return self.session.run(self.predicted_values, {self.prev_states: prev_states, self.states: states})

    def predict_softmax(self, prev_states, states):
        return self.session.run(self.softmax, {self.prev_states: prev_states, self.states: states})

    def train(self, prev_states, states, actions, returns, weights):
        self.session.run(self.training, {self.prev_states: prev_states, self.states: states, self.actions: actions, self.returns: returns, self.weights: weights})

    def save(self, path):
        self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='model_712', type=str, help="Network model to load.")
    parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
    parser.add_argument("--frame_skip", default=5, type=int, help="Repeat actions for given number of frames.")
    parser.add_argument("--frame_history", default=2, type=int, help="Number of past frames to stack together.")
    parser.add_argument("--render_each", default=1, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--alpha", default=None, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")

    parser.add_argument("--batch_size", default=32, type=int, help="Number of episodes to train on.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--padding", default="same", type=str)
    parser.add_argument("--buffer_size", default=100000, type=int)

    parser.add_argument("--replay_priority", default=0.6, type=float)
    parser.add_argument("--replay_beta", default=0.01, type=float)
    parser.add_argument("--replay_beta_final", default=1.0, type=float)

    parser.add_argument("--network_copy_steps", default=1000, type=int)

    args = parser.parse_args()

    # Create the environment
    env = car_racing_evaluator.environment()
    actions_mapping = [[-1, 0, 0], [0, 0, 0], [1, 0, 0],
                       [0, 1, 0], [0, 0, 1]]

    # Create network
    conv_layers = [[8, 3, 1], [0, 2, 2], [16, 3, 1], [0, 2, 2], [32, 3, 1], [0, 2, 2]]
    state_shape = [96, 96, 1]
    hidden_size = 32

    network_1 = Network(threads=args.threads)
    network_1.construct(args, conv_layers, hidden_size, state_shape, len(actions_mapping))

    network_2 = Network(threads=args.threads)
    network_2.construct(args, conv_layers, hidden_size, state_shape, len(actions_mapping))

    network_3 = Network(threads=args.threads)
    network_3.construct(args, conv_layers, hidden_size, state_shape, len(actions_mapping))

    network_4 = Network(threads=args.threads)
    network_4.construct(args, conv_layers, hidden_size, state_shape, len(actions_mapping))

    network_1.load('car_racing/model_712')
    network_2.load('car_racing/model_726')
    network_3.load('car_racing/model_776')
    network_4.load('car_racing/model_788')

    for episode in range(100):
        # Perform episode
        state, done = env.reset(True), False
        state = rgb2gray(state, state_shape)

        prev_state = state

        buffer = collections.deque(maxlen=5)
        for _ in range(5):
            buffer.append(state)

        while not done:
            if args.render_each and episode % args.render_each == 0:
                env.render()

            prev_state = buffer.popleft()

            q_values = 0.712*network_1.predict_softmax([prev_state], [state])[0]
            q_values += 0.726*network_2.predict_softmax([prev_state], [state])[0]
            q_values += 0.776*network_3.predict_softmax([prev_state], [state])[0]
            q_values += 0.788*network_4.predict_softmax([prev_state], [state])[0]

            action = np.argmax(q_values)

            buffer.append(state)

            state, reward, done, _ = env.step(actions_mapping[action], 1)
            state = rgb2gray(state, state_shape)
