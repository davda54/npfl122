#!/usr/bin/env python3
import numpy as np
import random

import cart_pole_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=100000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=10000, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.01, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # Initialize
    value_function = [[0] * env.actions] * env.states
    counter = [[0] * env.actions] * env.states

    for _ in range(args.episodes):

        # Perform a training episode
        state, done = env.reset(), False
        episode = []
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # select action
            if random.uniform(0.0, 1.0) < args.epsilon:
                action = random.randint(0, env.actions - 1)
            else:
                max_action = None; max_value = -float('inf')
                for action in range(env.actions):
                    value = value_function[state][action]
                    if value > max_value:
                        max_action = action
                        max_value = value
                action = max_action

            # make a step
            next_state, reward, done, _ = env.step(action)

            episode.append([state, action, reward])
            state = next_state

        G = 0
        for step in reversed(episode):
            G = args.gamma * G + step[2]
            counter[step[0]][step[1]] += 1
            value_function[step[0]][step[1]] += (G - value_function[step[0]][step[1]]) / counter[step[0]][step[1]]


    policy = [0] * env.states

    for state in range(env.states):
        max_action = None; max_value = -float('inf')
        for action in range(env.actions):
            value = value_function[state][action]
            if value > max_value:
                max_action = action
                max_value = value
        policy[state] = max_action


    # Perform last 100 evaluation episodes
    env.reset(start_evaluate = True)

    for _ in range(args.episodes):

        # Perform an evaluation episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = policy[state]
            state, reward, done, _ = env.step(action)
