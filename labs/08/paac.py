#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import gym_evaluator

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

            # TODO(reinforce_with_baseline): Compute `self.values`, starting with self.states and
            # - add a fully connected layer of size args.hidden_layer and ReLU activation
            # - add a fully connected layer with 1 output and no activation
            # - modify the result to have shape `[batch_size]` (you can use for example `[:, 0]`)
            hidden = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)
            self.values = tf.layers.dense(hidden, 1)[:, 0]

            # TODO: Compute `loss` as a sum of three losses:
            # - sparse softmax cross entropy of `self.actions` and `logits`,
            #   weighted by `self.returns - tf.stop_gradient(self.values)`.
            # - negative value of the distribution entropy (use `entropy` method of
            #   `tf.distributions.Categorical`) weighted by `args.entropy_regularization`.
            # - mean square error of the `self.returns` and `self.values`
            predict_loss = tf.losses.sparse_softmax_cross_entropy(self.actions, logits, weights=self.returns - tf.stop_gradient(self.values))
            baseline_loss = tf.losses.mean_squared_error(self.returns, self.values)
            loss = predict_loss + baseline_loss

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict_actions(self, states):
        return self.session.run(self.probabilities, {self.states: states})

    def predict_values(self, states):
        return self.session.run(self.values, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
    parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
    parser.add_argument("--evaluate_each", default=2500, type=int, help="Evaluate each number of batches.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")
    args = parser.parse_args()

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    # Initialize parallel workers by env.parallel_init
    states = env.parallel_init(args.workers)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Choose actions using network.predict_actions
            action_probabilities = network.predict_actions(states)
            actions = [np.random.choice(np.arange(env.actions), p=prob) for prob in action_probabilities]

            # TODO: Perform steps by env.parallel_steps
            next_states, rewards, done, _ = zip(*env.parallel_step(actions))

            # TODO: Compute return estimates by
            # - extracting next_states from steps
            # - computing value function approximation in next_states
            # - estimating returns by reward + (0 if done else args.gamma * next_state_value)
            next_returns = network.predict_values(next_states)
            returns = [rewards[i] + (not done[i])*next_returns[i] for i in range(args.workers)]

            # TODO: Train network using current states, chosen actions and estimated returns
            network.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                probabilities = network.predict_actions([state])[0]
                action = np.argmax(probabilities)
                state, _, done, _ = env.step(action)

        score = np.mean(env._episode_returns[-args.evaluate_for:])
        if score > 475:
            training = False

    while True:
        state, done = env.reset(True), False
        while not done:
            probabilities = network.predict_actions([state])[0]
            action = np.argmax(probabilities)
            state, _, done, _ = env.step(action)
