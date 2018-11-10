#!/usr/bin/env python3
import numpy as np
from numpy import inf
import random

import lunar_lander_evaluator

def greedy_action_1(Q, state):
    max_action = None
    max_q = -inf
    for a in range(env.actions):
        curr_q = Q[state, a]
        if curr_q > max_q:
            max_action = a
            max_q = curr_q

    return max_action

def greedy_action_2(Q_1, Q_2, state):
    max_action = None
    max_q = -inf
    for a in range(env.actions):
        curr_q = Q_1[state, a] + Q_2[state, a]
        if curr_q > max_q:
            max_action = a
            max_q = curr_q

    return max_action


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=1000, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.6, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.9999, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # The environment has env.states states and env.actions actions.

    # TODO: Implement a suitable RL algorithm.

    Q_1 = np.zeros([env.states, env.actions])
    Q_2 = np.zeros([env.states, env.actions])

    for _ in range(0):
        state, trajectory = env.expert_trajectory()

        for action, reward, next_state in trajectory:
            if random.uniform(0.0, 1.0) < 0.5:
                Q_1[state, action] += args.alpha*(reward + args.gamma*Q_2[next_state, greedy_action_1(Q_1, next_state)] - Q_1[state, action])
            else:
                Q_2[state, action] += args.alpha*(reward + args.gamma*Q_1[next_state, greedy_action_1(Q_2, next_state)] - Q_2[state, action])

            state = next_state

    for episode in range(args.episodes):

        # Perform a training episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            if random.uniform( 0.0, 1.0) < args.epsilon:
                action = random.randint(0, env.actions - 1)
            else:
                action = greedy_action_2(Q_1, Q_2, state)

            next_state, reward, done, _ = env.step(action)

            if random.uniform(0.0, 1.0) < 0.5:
                Q_1[state, action] += args.alpha * (reward + args.gamma * Q_2[next_state, greedy_action_1(Q_1, next_state)] - Q_1[state, action])
            else:
                Q_2[state, action] += args.alpha * (reward + args.gamma * Q_1[next_state, greedy_action_1(Q_2, next_state)] - Q_2[state, action])

            state = next_state

        args.epsilon *= 0.9995
        args.alpha *= 0.9995

        if episode % 100 == 0:
            print("epsilon: " + str(args.epsilon) + ", alpha: " + str(args.alpha))


    # Perform last 100 evaluation episodes
    policy = [0] * env.states

    for state in range(env.states):
        policy[state] = greedy_action_2(Q_1, Q_2, state)

    env.reset(start_evaluate=True)

    for _ in range(100):

        # Perform an evaluation episode
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            action = policy[state]
            state, reward, done, _ = env.step(action)