import gym
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

# Referenced https://medium.com/@tuzzer/follow-up-cart-pole-balancing-with-q-network-976d13f88d2f

class Episodes:
    def __init__(self):
        self.episodes = []
        self.max_episodes = 100000
        self.sample_size = 50

    def add(self, episode):
        self.episodes.append(episode)
        self.episodes = self.episodes[-self.max_episodes:]

    def sample(self):
        n = min(len(self.episodes), self.sample_size)
        idx = np.random.choice(len(self.episodes), n, replace=False)
        return [self.episodes[x] for x in idx]

class QNetwork:
    def __init__(self, state_dim, action_dim):
        self.q = Sequential()
        self.q.add(Dense(units=state_dim*3, activation='relu', input_dim=state_dim))
        self.q.add(Dense(units=state_dim*3, activation='relu'))
        self.q.add(Dense(units=action_dim, activation=None))
        self.q.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])

    def predict(self, state):
        return self.q.predict(state[:,-2:])

    def update(self, target, curr_state):
        return self.q.fit(np.array(curr_state), np.array(target), verbose=0)

class QCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.discount_rate = 0.99

        self.episodes = Episodes()

        state_dim = self.env.observation_space.shape[0] - 2 # Number of dims per layer
        action_dim = self.env.action_space.n
        self.q_network = QNetwork(state_dim, action_dim)

    def get_explore_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 50)))

    def get_learning_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 50)))
    
    def choose_action(self, state, explore_rate):
        return self.env.action_space.sample() if (np.random.random() < explore_rate) else np.argmax(self.q_network.predict(state))

    def update_network(self, learning_rate, verbose=False):
        episodes = self.episodes.sample()
        targets = []
        states = []
        for episode in episodes:
            curr_state, action, reward, next_state, done = episode
            target = self.q_network.predict(curr_state)
            if done:
                target[0][action] += learning_rate * (reward - target[0][action])
            else:
                target[0][action] += learning_rate * (reward + self.discount_rate * np.max(self.q_network.predict(next_state)) - target[0][action])
            targets.append(np.array(target).flatten())
            states.append(np.array(curr_state).flatten()[-2:])

        v = 1 if verbose else 0
        self.q_network.update(targets, states, v)

    def run(self):
        rewards = []
        for i in range(1000):
            curr_state = self.env.reset().reshape(1, -1)

            explore_rate = self.get_explore_rate(i)
            learning_rate = self.get_learning_rate(i)
            episode_rewards = 0

            for j in range(500):
                self.env.render()

                action = self.choose_action(curr_state, explore_rate)
                obs, reward, done, _ = self.env.step(action)

                next_state = obs.reshape(1, -1)
                episode = (curr_state, action, reward, next_state, done)
                self.episodes.add(episode)

                self.update_network(learning_rate)

                curr_state = next_state
                episode_rewards += reward

                if done:
                    print("Episode %d finished in %d timesteps. Explore rate: %f" % (i, j, explore_rate))
                    rewards.append(episode_rewards)
                    break

        self.env.close()
        print(rewards[-100:])
        print(np.mean(rewards[-100:]))
        print(np.max(rewards[-100:]))

if __name__ == "__main__":
    np.random.seed(0)
    solver = QCartPoleSolver()
    solver.run()
