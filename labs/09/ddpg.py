#!/usr/bin/env python3
import collections

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

    def construct(self, args, state_shape, action_components, action_lows, action_highs):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            self.next_states = tf.placeholder(tf.float32, [None] + state_shape)
            self.actions = tf.placeholder(tf.float32, [None, action_components])
            self.rewards = tf.placeholder(tf.float32, [None])
            self.done = tf.placeholder(tf.bool, [None])

            self.returns = tf.placeholder(tf.float32, [None])

            # Actor
            def actor(inputs):
                # TODO: Implement actor network, starting with `inputs` and returning
                # action_components values for each batch example. Usually, one
                # or two hidden layers are employed.
                #
                # Each action_component[i] should be mapped to range
                # [actions_lows[i]..action_highs[i]], for example using tf.nn.sigmoid
                # and suitable rescaling.
                actor_hidden = tf.layers.dense(inputs, args.hidden_layer, activation=tf.nn.relu)
                actor_hidden = tf.layers.dense(actor_hidden, args.hidden_layer, activation=tf.nn.relu)
                actor_hidden = tf.layers.dense(actor_hidden, action_components, activation=tf.nn.sigmoid)
                return actor_hidden*(action_highs - action_lows) + action_lows

            with tf.variable_scope("actor"):
                self.mus = actor(self.states)

            with tf.variable_scope("target_actor"):
                target_actions = actor(self.next_states)

            # Critic from given actions
            def critic(inputs, actions):
                # TODO: Implement critic network, starting with `inputs` and `actions`
                # and producing a vector of predicted returns. Usually, `inputs` are fed
                # through a hidden layer first, and then concatenated with `actions` and fed
                # through two more hidden layers, before computing the returns.
                input_hidden = tf.layers.dense(inputs, args.hidden_layer, activation=tf.nn.relu)
                critic_hidden = tf.concat([input_hidden, actions], 1)
                critic_hidden = tf.layers.dense(critic_hidden, args.hidden_layer, activation=tf.nn.relu)
                critic_hidden = tf.layers.dense(critic_hidden, args.hidden_layer, activation=tf.nn.relu)
                return tf.layers.dense(critic_hidden, 1)[:, 0]

            with tf.variable_scope("critic"):
                values_of_given = critic(self.states, self.actions)

            with tf.variable_scope("critic", reuse=True):
                values_of_predicted = critic(self.states, self.mus)

            with tf.variable_scope("target_critic"):
                self.target_values = critic(self.next_states, target_actions)

            # Update ops
            update_target_ops = []
            for target_var, var in zip(tf.global_variables("target_actor") + tf.global_variables("target_critic"),
                                       tf.global_variables("actor") + tf.global_variables("critic")):
                update_target_ops.append(target_var.assign((1.-args.target_tau) * target_var + args.target_tau * var))

            # TODO: Training
            # Define actor_loss and critic loss and then:
            # - train the critic (if required, using critic variables only,
            #     by using `var_list` argument of `Optimizer.minimize`)
            # - train the actor (if required, using actor variables only,
            #     by using `var_list` argument of `Optimizer.minimize`)
            # - update target network variables
            # You can group several operations into one using `tf.group`.

            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(args.learning_rate)

            returns = self.rewards + tf.where(self.done, tf.zeros_like(self.target_values), self.target_values)
            critic_loss = tf.losses.mean_squared_error(tf.stop_gradient(returns), values_of_given)
            critic_training = optimizer.minimize(critic_loss, global_step=global_step)

            actor_loss = -tf.reduce_mean(values_of_predicted)
            actor_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')
            actor_training = optimizer.minimize(actor_loss, var_list=actor_train_vars, global_step=global_step)

            self.training = tf.group(critic_training, actor_training)

            with tf.control_dependencies([self.training]):
                self.target_updating = tf.group(update_target_ops)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict_actions(self, states):
        return self.session.run(self.mus, {self.states: states})

    def train(self, states, next_states, actions, rewards, done):
        self.session.run([self.training, self.target_updating], {self.states: states, self.next_states: next_states, self.actions: actions, self.rewards: rewards, self.done: done})


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
    parser.add_argument("--evaluate_each", default=90, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate for number of batches.")
    parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
    parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=96, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")
    parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = gym_evaluator.GymEnvironment(args.env)
    assert len(env.action_shape) == 1
    action_lows, action_highs = map(np.array, env.action_ranges)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.action_shape[0], action_lows, action_highs)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=500000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(evaluating=False):
        rewards = 0
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 90:
                env.render()

            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_shape[0], 0., args.noise_theta, args.noise_sigma)

    while True:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                action = network.predict_actions([state])[0] + noise.sample()
                next_state, reward, done, _ = env.step(action)

                replay_buffer.append(Transition(state, action, reward, done, next_state))

                # If the replay_buffer is large enough, perform training
                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = zip(*[replay_buffer[i] for i in batch])
                    rewards = [r if r > -10 else -10 for r in rewards]

                    network.train(states, next_states, actions, rewards, dones)

                state = next_state

        # Evaluation
        returns = []
        for _ in range(args.evaluate_for):
            returns.append(evaluate_episode())

        average_return = np.mean(returns)
        print("Evaluation of {} episodes: {}".format(args.evaluate_for, np.mean(returns)))

        # if average_return > -180:
        #     while True:
        #         evaluate_episode(True)



