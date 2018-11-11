#!/usr/bin/env python3
#
# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b

import math
import numpy as np
import random

class MultiArmedBandits():
    def __init__(self, bandits, episode_length):
        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(np.random.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length
        #print("Initialized {}-armed bandit, maximum average reward is {}".format(bandits, np.max(self._bandits)))

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = np.random.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")

    parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
    parser.add_argument("--alpha", default=0.15, type=float, help="Learning rate to use (if applicable).")
    parser.add_argument("--c", default=1., type=float, help="Confidence level in ucb.")
    parser.add_argument("--epsilon", default=1/64, type=float, help="Exploration factor (if applicable).")
    parser.add_argument("--initial", default=1, type=float, help="Initial value function levels.")
    args = parser.parse_args()

    env = MultiArmedBandits(args.bandits, args.episode_length)

    average_rewards = []
    for episode in range(args.episodes):
        env.reset()

        q = np.full(args.bandits, args.initial, dtype=float)
        n = np.zeros(args.bandits)
        h = np.zeros(args.bandits)
        pi = np.zeros(args.bandits)

        average_rewards.append(0)
        t = 0
        done = False
        while not done:
            t += 1
            if args.mode == "greedy":

                if random.uniform(0, 1) < args.epsilon:
                    action = random.randint(0, args.bandits - 1)
                else:
                    action = np.argmax(q)

            elif args.mode == "ucb":

                action = np.argmax(q + args.c * np.sqrt(math.log(t) / (n + 1)))

            elif args.mode == "gradient":

                exp = np.exp(h)
                s = np.sum(exp)
                pi = exp / s

                action = np.random.choice(np.arange(0, args.bandits), p=pi)

            _, reward, done, _ = env.step(action)
            average_rewards[-1] += reward / args.episode_length

            n[action] += 1
            if args.alpha == 0 or args.mode == "ucb":
                q[action] += (reward - q[action]) / n[action]
            else:
                q[action] += args.alpha * (reward - q[action])

            h[action] += args.alpha * reward * (1 - pi[action])
            h[np.arange(len(h)) != action] -= args.alpha * reward * pi[np.arange(len(pi)) != action]

    # Print out final score as mean and variance of all obtained rewards.
    print("Final score: {}, variance: {}".format(np.mean(average_rewards), np.var(average_rewards)))