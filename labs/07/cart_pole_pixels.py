#!/usr/bin/env python3
#
# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

import numpy as np
import tensorflow as tf

import cart_pole_pixels_evaluator


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(
            graph=graph,
            config=tf.ConfigProto(
                inter_op_parallelism_threads=threads,
                intra_op_parallelism_threads=threads,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        )

    def construct(self, args, convolution, hidden_size, state_shape, num_actions):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.int32, [None])
            self.returns = tf.placeholder(tf.float32, [None])

            input = tf.image.crop_to_bounding_box(self.states, 0, 20, 40, 40)
            action_output = input
            for filters, kernel, stride in convolution:
                if filters == 0:
                    action_output = tf.layers.max_pooling2d(inputs=action_output, pool_size=[kernel, kernel],
                                                            strides=stride)
                else:
                    action_output = tf.layers.conv2d(inputs=action_output, filters=filters,
                                                     kernel_size=[kernel, kernel],
                                                     strides=stride, padding=args.padding)
                    action_output = tf.nn.relu(action_output)
            action_output = tf.layers.flatten(action_output)
            action_output = tf.layers.dense(action_output, hidden_size, activation=tf.nn.relu)
            logits = tf.layers.dense(action_output, num_actions)
            self.predictions = tf.nn.softmax(logits, axis=1)

            predict_loss = tf.losses.sparse_softmax_cross_entropy(self.actions, logits,
                                                                  weights=self.returns)

            # Saver for the inference network
            self.saver = tf.train.Saver()

            # TODO: Training using operation `self.training`.
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(predict_loss,
                                                                                global_step=global_step,
                                                                                name="training")  # + baseline_loss

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.predictions, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

    def save(self, path):
        self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import time

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=True, type=bool, help="Checkpoint path.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

    parser.add_argument("--checkpoint_every", default=32, type=int, help="Period with which the checkpoint is stored.")
    parser.add_argument("--batch_size", default=32, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=200000, type=int, help="Training episodes.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--padding", default="same", type=str, help="Learning rate.")

    args = parser.parse_args()
    timestamp = time.strftime("%H-%M-%S")

    # Create the environment
    env = cart_pole_pixels_evaluator.environment()

    # Construct the network

    conv_layers = [[8, 3, 1], [0, 2, 2], [16, 3, 1], [0, 2, 2], [32, 3, 1], [0, 2, 2]]
    hidden_size = 32

    network = Network(threads=args.threads)
    network.construct(args, conv_layers, hidden_size, env.state_shape, env.actions)

    network_1 = Network(threads=args.threads)
    network_1.construct(args, conv_layers, hidden_size, env.state_shape, env.actions)

    network_2 = Network(threads=args.threads)
    network_2.construct(args, conv_layers, hidden_size, env.state_shape, env.actions)

    # Load the checkpoint if required
    if args.evaluate:
        # Try extract it from embedded_data
        try:
            import embedded_data

            embedded_data.extract()
        except:
            pass

        network_1.load('cart_pole_pixels/davda_model_500')
        network_2.load('cart_pole_pixels/davda_model_499.51')

        for _ in range(100):
            # Perform episode
            state, done = env.reset(start_evaluate=True), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action_probabilities = network_1.predict([state])[0] + network_2.predict([state])[0]
                action = np.argmax(action_probabilities)

                state, _, done, _ = env.step(action)

    else:
        episode = 0
        best_score = 0
        batch_states, batch_actions, batch_returns = [], [], []
        while True:
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action_probabilities = network.predict([state])[0]
                action = np.random.choice(np.arange(env.actions), p=action_probabilities)

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            returns = [sum(rewards[i:]) for i in range(len(states))]
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


            episode += 1

            if episode > 500 and episode % 200 == 0:
                for _ in range(50):
                    state, done = env.reset(), False
                    while not done:
                        action = np.argmax(network.predict([state])[0])

                        next_state, reward, done, _ = env.step(action)

                        state = next_state

                score = np.mean(env._episode_returns[-50:])
                if score > best_score or score > 499:
                    best_score = score
                    checkpoint_path = "cart_pole_pixels/model_{s}_{t}".format(s=int(best_score), t=timestamp)
                    print("Best score improved to {s}, saving {p}".format(s=best_score, p=checkpoint_path))
                    network.save(checkpoint_path)
