#!/usr/bin/env python3
import numpy as np
import random
import itertools
from numpy import inf

import mountain_car_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.4, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    parser.add_argument("--n", default=4, type=int)
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W_1 = np.zeros([env.weights, env.actions])
    W_2 = np.zeros([env.weights, env.actions])

    for epsilon in [0.2]:
       for alpha in [0.2]:

            alpha = alpha / args.tiles
            epsilon = args.epsilon

            best_score = -1000

            for episode in range(args.episodes):

                # Perform a training episode
                state = env.reset()

                if random.uniform(0.0, 1.0) < args.epsilon:
                    action = random.randint(0, env.actions - 1)
                else:
                    action = np.argmax(W_1[state].sum(axis=0) + W_2[state].sum(axis=0))

                states = [state]
                actions = [action]
                rewards = [0]

                T = inf
                for t in itertools.count():

                    if t < T:
                        if args.render_each and env.episode and env.episode % args.render_each == 0:
                            env.render()

                        state, reward, done, _ = env.step(actions[t])

                        states.append(state)
                        rewards.append(reward)

                        if done:
                            T = t + 1
                        else:
                            if random.uniform(0.0, 1.0) < args.epsilon:
                                actions.append(random.randint(0, env.actions - 1))
                            else:
                                actions.append(np.argmax(W_1[state].sum(axis=0) + W_2[state].sum(axis=0)))

                    tau = t + 1 - args.n

                    if tau >= 0:
                        if random.uniform(0.0, 1.0) < 0.5:
                            W_a = W_1; W_b = W_2
                        else:
                            W_a = W_2; W_b = W_1

                        if t + 1 >= T:
                            G = rewards[T]
                        else:
                            G = rewards[t + 1] + W_b[states[t + 1], np.argmax(W_a[states[t + 1]].sum(axis=0))].sum()

                        for k in range(min(t, T - 1), tau + 1, -1):
                            greedy_action = np.argmax(W_a[states[k]].sum(axis=0))
                            G = rewards[k] + args.gamma * (G if greedy_action == actions[k] else W_b[states[k], greedy_action].sum())

                        W_a[states[tau], actions[tau]] += alpha * (G - W_a[states[tau], actions[tau]].sum())

                    if tau == T - 1:
                        break

                if args.epsilon_final:
                    epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
                if args.alpha_final:
                    alpha = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

                if episode % 100 == 0:

                    mean_return = 0
                    for eval_episode in range(100):
                        state, done = env.reset(False), False
                        ret = 0
                        while not done:
                            # TODO: choose action as a greedy action
                            action = np.argmax(W_1[state, :].sum(0) + W_2[state, :].sum(0))
                            state, reward, done, _ = env.step(action)

                            ret += reward

                        mean_return = (mean_return * eval_episode + ret) / (eval_episode + 1)

                    if mean_return > best_score:
                        best_score = mean_return
                        best_W = W_1 + W_2

                    # print(mean_return)

                    if mean_return > -105:
                        break

            # print("alpha: " + str(alpha * args.tiles) + "\tepsilon: " + str(epsilon) + "\tscore:" + str(best_score))

    # Perform the final evaluation episodes

    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(best_W[state, :].sum(0))
            state, reward, done, _ = env.step(action)