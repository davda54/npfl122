#!/usr/bin/env python3
import numpy as np
from numpy import inf
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

    parser.add_argument("--alpha", default=0.6, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment()

    # TODO: Implement Q-learning RL algorithm.
    q = [[0] * env.actions for _ in range(env.states)]

    # The overall structure of the code follows.

    for _ in range(args.episodes):
        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if random.uniform(0.0, 1.0) < args.epsilon:
                action = random.randint(0, env.actions - 1)
            else:
                max_action = None; max_q = -inf
                for a in range(env.actions):
                    curr_q = q[state][a]
                    if curr_q > max_q:
                        max_action = a
                        max_q = curr_q
                action = max_action

            next_state, reward, done, _ = env.step(action)

            max_action = None; max_q = -inf
            for a in range(env.actions):
                curr_q = q[next_state][a]
                if curr_q > max_q:
                    max_action = a
                    max_q = curr_q

            q[state][action] += args.alpha * (reward + args.gamma * q[next_state][max_action] - q[state][action])

            state = next_state
            args.epsilon *= 0.99999
            args.alpha *= 0.999996

    policy = [0] * env.states

    for state in range(env.states):
        max_action = None; max_value = -float('inf')
        for action in range(env.actions):
            value = q[state][action]
            if value > max_value:
                max_action = action
                max_value = value
        policy[state] = max_action

    env.reset(start_evaluate=True)

    for _ in range(100):

        # Perform an evaluation episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = policy[state]
            state, reward, done, _ = env.step(action)
