#!/usr/bin/env python3
import numpy as np
import random
import itertools
from numpy import inf

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
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=1000, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.3, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
    parser.add_argument("--n", default=4, type=int)

    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # The environment has env.states states and env.actions actions.

    # TODO: Implement a suitable RL algorithm.

    Q_1 = np.zeros([env.states, env.actions])
    Q_2 = np.zeros([env.states, env.actions])

    for _ in range(100):
        state, trajectory = env.expert_trajectory()

        for action, reward, next_state in trajectory:
            if random.uniform(0.0, 1.0) < 0.5:
                Q_1[state, action] += args.alpha*(reward + args.gamma*Q_2[next_state, greedy_action_1(Q_1, next_state)] - Q_1[state, action])
            else:
                Q_2[state, action] += args.alpha*(reward + args.gamma*Q_1[next_state, greedy_action_1(Q_2, next_state)] - Q_2[state, action])

            state = next_state

    for episode in range(args.episodes):

        # Perform a training episode
        state = env.reset()

        if random.uniform(0.0, 1.0) < args.epsilon:
            action = random.randint(0, env.actions - 1)
        else:
            action = greedy_action_2(Q_1, Q_2, state)

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
                        actions.append(greedy_action_2(Q_1, Q_2, state))

            tau = t + 1 - args.n

            if tau >= 0:
                if random.uniform(0.0, 1.0) < 0.5:
                    Q_a = Q_1; Q_b = Q_2
                else:
                    Q_a = Q_2; Q_b = Q_1

                if t + 1 >= T:
                    G = rewards[T]
                else:
                    G = rewards[t + 1] + args.gamma * Q_b[states[t + 1], greedy_action_1(Q_a, states[t + 1])]

                for k in range(min(t, T - 1), tau + 1, -1):
                    greedy_action = greedy_action_1(Q_a, states[k])
                    G = rewards[k] + args.gamma * (G if greedy_action == actions[k] else Q_b[states[k], greedy_action])

                Q_a[states[tau], actions[tau]] += args.alpha * (G - Q_a[states[tau], actions[tau]])

            if tau == T - 1:
                break


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