#!/usr/bin/env python3
import numpy as np
import random

import mountain_car_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.8, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.05, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.4, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.005, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles, verbose=False)

    # Implement Q-learning RL algorithm, using linear approximation.
    W_1 = np.zeros([env.weights, env.actions])
    W_2 = np.zeros([env.weights, env.actions])

    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    best_score = -1000

    for episode in range(args.episodes):
        # Perform a training episode
        state, done = env.reset(False), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # TODO: Choose `action` according to epsilon-greedy strategy
            if random.uniform(0.0, 1.0) < epsilon:
                action = random.randint(0, env.actions - 1)
            else:
                action = np.argmax(W_1[state, :].sum(0) + W_2[state, :].sum(0))

            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values
            if random.uniform(0.0, 1.0) < 0.5:
                W_1[state, action] += alpha * (reward + W_2[next_state, np.argmax(W_1[next_state, :].sum(0))] - W_1[state, action])
            else:
                W_2[state, action] += alpha * (reward + W_1[next_state, np.argmax(W_2[next_state, :].sum(0))] - W_2[state, action])

            state = next_state
            if done:
                break

        if args.epsilon_final:
            epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
        if args.alpha_final:
            alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

        if episode > 2000 and episode % 100 == 0:

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

            print(mean_return)

            if mean_return > -100:
                break

    # Perform the final evaluation episodes

    while True:
        state, done = env.reset(True), False
        while not done:
            action = np.argmax(best_W[state, :].sum(0))
            state, reward, done, _ = env.step(action)