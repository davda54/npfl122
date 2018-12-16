#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

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
            self.returns = tf.placeholder(tf.float32, [None])

            # TODO(reinforce): Start with self.states and
            # - add a fully connected layer of size args.hidden_layer and ReLU activation
            # - add a fully connected layer with num_actions and no activation, computing `logits`
            # - compute `self.probabilities` as tf.nn.softmax of `logits`
            hidden = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)
            logits = tf.layers.dense(hidden, num_actions)
            self.probabilities = tf.nn.softmax(logits, axis=1)

            # TODO: Compute `baseline`, by starting with `self.states` and
            # - add a fully connected layer of size args.hidden_layer and ReLU activation
            # - add a fully connected layer with 1 output and no activation
            # - modify the result to have shape `[batch_size]` (you can use for example `[:, 0]`)
            hidden = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)
            baseline = tf.layers.dense(hidden, 1)[:, 0]

            # TODO: Compute `loss` as a sum of two losses:
            # - sparse softmax cross entropy of `self.actions` and `logits`,
            #   weighted by `self.returns - baseline`. You should not backpropagate
            #   gradient into `baseline` by using `tf.stop_gradient(baseline)`.
            # - mean square error of the `self.returns` and `baseline`
            predict_loss = tf.losses.sparse_softmax_cross_entropy(self.actions, logits, weights=self.returns - tf.stop_gradient(baseline))
            baseline_loss = tf.losses.mean_squared_error(self.returns, baseline)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(predict_loss + baseline_loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.probabilities, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    # np.random.seed(13)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=256, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    batch_states, batch_actions, batch_returns = [], [], []

    # Training
    for _ in range(args.episodes):
        # Perform episode
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()
            # TODO: Compute action probabilities using `network.predict` and current `state`
            action_probabilities = network.predict([state])[0]
            # TODO: Choose `action` according to `probabilities` distribution (np.random.choice can be used)
            action = np.random.choice(np.arange(env.actions), p=action_probabilities)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

            # TODO: Compute returns by summing rewards (with discounting)
        returns = [sum(rewards[i:]) for i in range(len(states))]
        # TODO: Add states, actions and returns to the training batch
        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_returns.extend(returns)

        # Train using the generated batch
        while len(batch_states) >= args.batch_size:
            network.train(np.array(batch_states[:args.batch_size - 1]),
                          np.array(batch_actions[:args.batch_size - 1]),
                          np.array(batch_returns[:args.batch_size - 1]))
            batch_states = batch_states[args.batch_size:]
            batch_actions = batch_actions[args.batch_size:]
            batch_returns = batch_returns[args.batch_size:]

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # TODO: Compute action `probabilities` using `network.predict` and current `state`
            probabilities = network.predict([state])[0]

            # Choose greedy action this time
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)