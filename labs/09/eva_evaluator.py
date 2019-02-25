#!/usr/bin/env python3
#
# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b

# import numpy as np
# import gym
# import _pickle as pickle
# import random
#
# class Model:
#
#     def __init__(self, input_size, output_size, hidden_layer):
#         self.weights = [np.random.randn(input_size, hidden_layer) / np.sqrt(input_size),
#                         #np.random.randn(hidden_layer, hidden_layer) / np.sqrt(hidden_layer),
#                         np.random.randn(hidden_layer, output_size) / np.sqrt(hidden_layer)]
#
#         self.exploration = args.exploration
#
#     def predict(self, state):
#         input_layer = np.matmul(state, self.weights[0])
#         input_layer = np.tanh(input_layer)
#
#         #hidden_layer = np.matmul(input_layer, self.weights[1])
#         #hidden_layer = np.tanh(hidden_layer)
#
#         output_layer = np.matmul(input_layer, self.weights[1])
#         output_layer = np.tanh(output_layer)
#
#         return output_layer
#
#     def load(self, path):
#         with open(path, 'rb') as fp:
#             self.weights = pickle.load(fp)
#
#     def save(self, path):
#         with open(path, 'wb') as fp:
#             pickle.dump(self.weights, fp)
#
#     def get_reward(self, env):
#         total_reward = 0.0
#
#         state, done = env.reset(), False
#         for _ in range(600):
#             self.exploration /= args.exploration_decay
#
#             #if random.random() < self.exploration:
#             #    action = env._env.action_space.sample()
#             #else:
#             action = self.predict(state)
#
#             state, reward, done, _ = env.step(action)
#             total_reward += reward if reward > -10 else -10
#
#             if done: return total_reward
#
#         return total_reward
#
#
# def evaluate(env, model, episodes):
#     avg_reward = 0
#     for episode in range(episodes):
#         total_reward = 0
#         state, done = env.reset(), False
#
#         while not done:
#             if episode % args.render_each == 0: env.render()
#
#             action = model.predict(state)
#             state, reward, done, _ = env.step(action)
#             total_reward += reward
#
#         avg_reward = (episode*avg_reward + total_reward) / (episode + 1)
#
#     print("EVALUATION REWARD: " + str(avg_reward))
#     return avg_reward
#
#
# def train(env, model, experimental_model, episodes):
#     for episode in range(episodes):
#
#         population = [[np.random.randn(*w.shape)*args.sigma for w in model.weights] for _ in range(args.population)]
#
#         rewards = []
#         for p in population:
#             experimental_model.weights = [model.weights[i] + noise for i, noise in enumerate(p)]
#             rewards.append(experimental_model.get_reward(env))
#         rewards = np.array(rewards)
#
#         rewards = (rewards - rewards.mean()) / rewards.std()
#         alpha = args.lr / (args.population * args.sigma)
#         for layer in range(len(model.weights)):
#             noise = np.array([p[layer] for p in population])
#             model.weights[layer] += alpha * np.dot(noise.T, rewards).T
#
#         args.lr *= args.lr_decay
#
#         print(str(episode) + ",\t" + str(model.get_reward(env)))
#
#
# if __name__ == "__main__":
#     # Fix random seed
#     np.random.seed(42)
#
#     # Parse arguments
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
#     parser.add_argument("--path", default="./eva_walker/model_299.7625022122523", type=str, help="Path to model.")
#     parser.add_argument("--population", default=512, type=int, help="Evaluate each number of episodes.")
#     parser.add_argument("--sigma", default=0.01, type=float, help="Exploration sigma.")
#     parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
#     parser.add_argument("--lr_decay", default=0.999, type=float, help="Learning rate.")
#     parser.add_argument("--hidden_layer", default=100, type=int, help="Size of hidden layer.")
#     parser.add_argument("--render_each", default=10, type=int, help="Render some episodes.")
#     parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
#     parser.add_argument("--episodes", default=10000, type=int)
#     parser.add_argument("--train_for", default=5, type=int, help="Train for number of episodes.")
#     parser.add_argument("--evaluate_for", default=25, type=int, help="Evaluate for number of episodes.")
#     parser.add_argument("--exploration", default=0.0, type=float)
#     parser.add_argument("--exploration_decay", default=0.99, type=float)
#
#     args = parser.parse_args()
#     env = gym.make(args.env)
#     #env = gym_evaluator.GymEnvironment(args.env)
#     #assert len(env.action_shape) == 1
#     #assert len(env.state_shape) == 1
#
#     model = Model(24, 4, args.hidden_layer)
#     experimental_model = Model(24, 4, args.hidden_layer)
#
#     model.load(args.path)
#
#     best_return = 289
#
#     for episode in range(args.episodes // (args.train_for + args.evaluate_for)):
#         train(env, model, experimental_model, args.train_for)
#         ret = evaluate(env, model, args.evaluate_for)
#
#         if ret > best_return:
#             best_return = ret
#             model.save("./eva_walker/model_" + str(ret))


import numpy as np
import gym_evaluator
import _pickle as pickle
import embedded_data

class Model:

    def __init__(self, input_size, output_size, hidden_layer):
        self.weights = [np.random.randn(input_size, hidden_layer) / np.sqrt(input_size),
                        np.random.randn(hidden_layer, output_size) / np.sqrt(hidden_layer)]

    def predict(self, state):
        input_layer = np.matmul(state, self.weights[0])
        input_layer = np.tanh(input_layer)

        output_layer = np.matmul(input_layer, self.weights[1])
        output_layer = np.tanh(output_layer)

        return output_layer

    def load(self, path):
        with open(path, 'rb') as fp:
            self.weights = pickle.load(fp)


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(1)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./model_299.887518416494", type=str, help="Path to model.")
    parser.add_argument("--env", default="BipedalWalker-v2", type=str, help="Environment.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of hidden layer.")
    args = parser.parse_args()

    embedded_data.extract()

    env = gym_evaluator.GymEnvironment(args.env)

    model = Model(24, 4, args.hidden_layer)
    model.load(args.path)

    while True:
        state, done = env.reset(start_evaluate=True), False

        while not done:
            action = model.predict(state)
            state, reward, done, _ = env.step(action)
