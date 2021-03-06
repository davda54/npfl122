#!/usr/bin/env python3
import collections

import numpy as np
import tensorflow as tf
import random

import cart_pole_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None])
            self.q_values = tf.placeholder(tf.float32, [None])

            # Compute the q_values
            hidden = self.states
            for _ in range(args.hidden_layers):
                hidden = tf.layers.dense(hidden, args.hidden_layer_size, activation=tf.nn.relu)
            self.predicted_values = tf.layers.dense(hidden, num_actions)

            # Training
            loss = tf.losses.mean_squared_error(self.q_values, tf.boolean_mask(self.predicted_values, tf.one_hot(self.actions, num_actions)))
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def copy_variables_from(self, other):
        for variable, other_variable in zip(self.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                                            other.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
            variable.load(other_variable.eval(other.session), self.session)

    def predict(self, states):
        return self.session.run(self.predicted_values, {self.states: states})

    def train(self, states, actions, q_values):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.q_values: q_values})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(113)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes for epsilon decay.")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
    parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    best_network = Network(threads=args.threads)
    best_network.construct(args, env.state_shape, env.actions)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=100000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    evaluating = False
    epsilon = args.epsilon
    best_score = 0

    for episode in range(args.episodes):
        # Perform episode
        state, done = env.reset(False), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            if random.uniform(0.0, 1.0) < epsilon:
                action = random.randint(0, env.actions - 1)
            else:
                action = np.argmax(network.predict([state])[0])

            next_state, reward, done, _ = env.step(action)

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            if len(replay_buffer) > args.batch_size:
                batch = [replay_buffer[random.randint(0, len(replay_buffer) - 1)] for _ in range(args.batch_size)]

                states = [t.state for t in batch]
                actions = [t.action for t in batch]
                predictions = network.predict([t.next_state for t in batch])

                q_values = np.zeros(args.batch_size)
                for i in range(args.batch_size):
                    q_values[i] = batch[i].reward + (0 if batch[i].done else np.max(predictions[i]))

                network.train(states, actions, q_values)

            state = next_state

        if args.epsilon_final:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))

        if episode > 100 and episode % 50 == 0:

            mean_return = 0
            for eval_episode in range(100):
                state, done = env.reset(False), False
                ret = 0
                while not done:
                    action = np.argmax(network.predict([state])[0])
                    state, reward, done, _ = env.step(action)

                    ret += reward

                mean_return = (mean_return * eval_episode + ret) / (eval_episode + 1)

            if mean_return > best_score:
                best_score = mean_return
                best_network.copy_variables_from(network)

            #if mean_return > 450:
            #    break

    # evaluation

    while True:
        state, done = env.reset(True), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            action = np.argmax(best_network.predict([state])[0])
            state, _, done, _ = env.step(action)
