#!/usr/bin/env python3
import numpy as np
import gym

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    args = parser.parse_args()

    # Create the environment
    env = gym.make("FrozenLake-v0")
    env.seed(42)
    states = env.observation_space.n
    actions = env.action_space.n

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(states)
    C = np.zeros(states)

    for _ in range(args.episodes):
        state, done = env.reset(), False

        # Generate episode
        episode = []
        while not done:
            action = np.random.choice(actions)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # TODO: Update V using weighted importance sampling.

        G = 0
        W = 1
        for step in episode[::-1]:
            if step[1] not in [1,2]: break;
            G += step[2]
            C[step[0]] += W
            V[step[0]] += W / C[step[0]] * (G - V[step[0]])
            W *= 2

    # Print the final value function V
    for row in V.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))
