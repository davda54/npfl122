import numpy as np
import tensorflow as tf
import gym
import _pickle as pickle

class Mengele:

    def __init__(self, input_size, output_size, hidden_layer, evicka, kun):
        self.weights = [np.random.randn(input_size, hidden_layer) / np.sqrt(input_size),
                        np.random.randn(hidden_layer) / np.sqrt(input_size),
                        #np.random.randn(hidden_layer, hidden_layer) / np.sqrt(hidden_layer),
                        #np.random.randn(hidden_layer) / np.sqrt(hidden_layer),
                        np.random.randn(hidden_layer, output_size) / np.sqrt(hidden_layer),
                        np.random.randn(output_size) / np.sqrt(hidden_layer)]

        self.evicka = evicka
        self.kun = kun



    def predict(self, state):
        input = state / np.linalg.norm(state)

        hidden_layer = np.matmul(input, self.weights[0]) + self.weights[1]
        hidden_layer = np.tanh(hidden_layer)

        #hidden_layer = np.matmul(hidden_layer, self.weights[2]) + self.weights[3]
        #hidden_layer = np.tanh(hidden_layer)

        output_layer = np.matmul(hidden_layer, self.weights[2]) + self.weights[3]
        output_layer = 1 / (1 + np.exp(-output_layer))

        action = output_layer * self.evicka.predict(state) + (1 - output_layer) * self.kun.predict(state)
        return action

    def load(self, path):
        with open(path, 'rb') as fp:
            self.weights = pickle.load(fp)

    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def get_reward(self, env):
        total_reward = 0.0

        state, done = env.reset(), False
        for _ in range(1000):
            action = self.predict(state)

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done: return total_reward

        return total_reward

class Evicka:

    def __init__(self, path):
        self.load(path)

    def predict(self, state):
        input_layer = np.matmul(state, self.weights[0])
        input_layer = np.tanh(input_layer)

        output_layer = np.matmul(input_layer, self.weights[1])
        output_layer = np.tanh(output_layer)

        return output_layer

    def load(self, path):
        with open(path, 'rb') as fp:
            self.weights = pickle.load(fp)


class Kun:
    def __init__(self, env, path):
        action_lows, action_highs = map(np.array, (list(env.action_space.low), list(env.action_space.high)))

        self.network = Network(threads=1)
        self.network.construct(args, 24, 4, action_lows, action_highs)
        self.network.load(path)

    def predict(self, state):
        return self.network.predict_actions([state])[0]


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, action_components, action_lows, action_highs):
        with self.session.graph.as_default():
            self.states = tf.placeholder(tf.float32, [None, state_shape])
            self.next_states = tf.placeholder(tf.float32, [None, state_shape])
            self.actions = tf.placeholder(tf.float32, [None, action_components])
            self.rewards = tf.placeholder(tf.float32, [None])
            self.dones = tf.placeholder(tf.bool, [None])

            # Actor
            def actor(inputs):
                # TODO: Implement actor network, starting with `inputs` and returning
                # action_components values for each batch example. Usually, one
                # or two hidden layers are employed.
                #
                # Each action_component[i] should be mapped to range
                # [actions_lows[i]..action_highs[i]], for example using tf.nn.sigmoid
                # and suitable rescaling.
                hidden = tf.layers.dense(inputs, 180, activation=tf.nn.relu)
                hidden = tf.layers.dense(hidden, 180, activation=tf.nn.relu)
                output = tf.layers.dense(hidden, action_components, activation=tf.nn.sigmoid)
                return output * (action_highs - action_lows) + action_lows

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
                hidden = tf.layers.dense(inputs, 180, activation=tf.nn.relu)
                hidden = tf.concat([hidden, actions], 1)
                hidden = tf.layers.dense(hidden, 180, activation=tf.nn.relu)
                hidden = tf.layers.dense(hidden, 180, activation=tf.nn.relu)
                return tf.layers.dense(hidden, 1)[:, 0]

            with tf.variable_scope("critic1"):
                values_of_given1 = critic(self.states, self.actions)

            with tf.variable_scope("critic1", reuse=True):
                values_of_predicted = critic(self.states, self.mus)

            with tf.variable_scope("critic2"):
                values_of_given2 = critic(self.states, self.actions)

            with tf.variable_scope("target_critic1"):
                target1_values = critic(self.next_states, target_actions)

            with tf.variable_scope("target_critic2"):
                target2_values = critic(self.next_states, target_actions)

            update_critic_ops = []
            for target_var, var in zip(tf.global_variables("target_critic1") + tf.global_variables("target_critic2"),
                                       tf.global_variables("critic1") + tf.global_variables("critic2")):
                update_critic_ops.append(target_var.assign((1. - 0.005) * target_var + 0.005 * var))

            update_actor_ops = []
            for target_var, var in zip(tf.global_variables("target_actor"),
                                       tf.global_variables("actor")):
                update_actor_ops.append(target_var.assign((1. - 0.005) * target_var + 0.005 * var))

            # TODO: Training
            # Define actor_loss and critic loss and then:
            # - train the critic (if required, using critic variables only,
            #     by using `var_list` argument of `Optimizer.minimize`)
            # - train the actor (if required, using actor variables only,
            #     by using `var_list` argument of `Optimizer.minimize`)
            # - update target network variables
            # You can group several operations into one using `tf.group`.
            actor_loss = -tf.reduce_mean(values_of_predicted)

            returns = self.rewards + tf.where(self.dones, tf.zeros_like(target1_values), tf.minimum(target1_values, target2_values))
            critic1_loss = tf.losses.mean_squared_error(tf.stop_gradient(returns), values_of_given1)
            critic2_loss = tf.losses.mean_squared_error(tf.stop_gradient(returns), values_of_given2)

            actor_global_step = tf.Variable(0, trainable=False, name='actor_step')
            training_actor = tf.train.AdamOptimizer(0.001).minimize(actor_loss, global_step=actor_global_step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor'))

            critic1_global_step = tf.Variable(0, trainable=False, name='critic1_step')
            critic2_global_step = tf.Variable(0, trainable=False, name='critic2_step')
            training_critic1 = tf.train.AdamOptimizer(0.001).minimize(critic1_loss, global_step=critic1_global_step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic1'))
            training_critic2 = tf.train.AdamOptimizer(0.001).minimize(critic2_loss, global_step=critic2_global_step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic2'))

            self.training = tf.group(training_actor, training_critic1, training_critic2)
            self.training_critics = tf.group(training_critic1, training_critic2)

            with tf.control_dependencies([self.training]):
                self.paining = tf.group(update_critic_ops, update_actor_ops)

            with tf.control_dependencies([self.training_critics]):
                self.paining_critics = tf.group(update_critic_ops)

            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict_actions(self, states):
        return self.session.run(self.mus, {self.states: states})

    def train_full(self, states, next_states, actions, dones, rewards):
        self.session.run([self.training, self.paining], {self.states: states, self.next_states: next_states, self.actions: actions, self.rewards: rewards, self.dones: dones})

    def train_critic(self, states, next_states, actions, dones, rewards):
        self.session.run([self.training_critics, self.paining_critics], {self.states: states, self.next_states: next_states, self.actions: actions, self.rewards: rewards, self.dones: dones})

    def save(self, path):
        self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

    def load(self, path):
        self.saver.restore(self.session, path)

def evaluate(env, model, episodes):
    avg_reward = 0
    for episode in range(episodes):
        total_reward = 0
        state, done = env.reset(), False

        while not done:
            if episode % args.render_each == 0: env.render()

            action = model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        avg_reward = (episode*avg_reward + total_reward) / (episode + 1)

    print("EVALUATION REWARD: " + str(avg_reward))
    return avg_reward


def train(env, model, experimental_model, episodes):
    for episode in range(episodes):

        population = [[np.random.randn(*w.shape)*args.sigma for w in model.weights] for _ in range(args.population)]

        rewards = []
        for p in population:
            experimental_model.weights = [model.weights[i] + noise for i, noise in enumerate(p)]
            rewards.append(experimental_model.get_reward(env))
        rewards = np.array(rewards)

        rewards = (rewards - rewards.mean()) / rewards.std()
        alpha = args.lr / (args.population * args.sigma)
        for layer in range(len(model.weights)):
            noise = np.array([p[layer] for p in population])
            model.weights[layer] += alpha * np.dot(noise.T, rewards).T

        args.lr *= args.lr_decay

        print(str(episode) + ",\t" + str(model.get_reward(env)))


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
    parser.add_argument("--evicka_path", default="./eva_walker/model_299.7058436320094", type=str, help="Path to model.")
    parser.add_argument("--kun_path", default="./kun/model_297.40549516797097", type=str, help="Path to model.")
    parser.add_argument("--population", default=256, type=int, help="Evaluate each number of episodes.")
    parser.add_argument("--sigma", default=0.02, type=float, help="Exploration sigma.")
    parser.add_argument("--lr", default=0.03, type=float, help="Learning rate.")
    parser.add_argument("--lr_decay", default=0.999, type=float, help="Learning rate.")
    parser.add_argument("--hidden_layer", default=64, type=int, help="Size of hidden layer.")
    parser.add_argument("--render_each", default=10, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--episodes", default=10000, type=int)
    parser.add_argument("--train_for", default=10, type=int, help="Train for number of episodes.")
    parser.add_argument("--evaluate_for", default=25, type=int, help="Evaluate for number of episodes.")
    parser.add_argument("--exploration", default=0.0, type=float)
    parser.add_argument("--exploration_decay", default=0.99, type=float)

    args = parser.parse_args()
    env = gym.make(args.env)
    #env = gym_evaluator.GymEnvironment(args.env)
    #assert len(env.action_shape) == 1
    #assert len(env.state_shape) == 1

    kun = Kun(env, args.kun_path)
    evicka = Evicka(args.evicka_path)

    model = Mengele(24, 4, args.hidden_layer, evicka, kun)
    experimental_model = Mengele(24, 4, args.hidden_layer, evicka, kun)

    best_return = 0

    for episode in range(args.episodes // (args.train_for + args.evaluate_for)):
        train(env, model, experimental_model, args.train_for)
        ret = evaluate(env, model, args.evaluate_for)

        if ret > best_return:
            best_return = ret
            model.save("./mengele/model_" + str(ret))
