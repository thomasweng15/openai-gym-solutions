import gym
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Referenced https://medium.com/@tuzzer/follow-up-cart-pole-balancing-with-q-network-976d13f88d2f


class QNetwork:

    def __init__(self, state_dim, action_dim):
        self.q = Sequential()
        self.q.add(Dense(units=state_dim*3, activation='relu', input_dim=state_dim))
        self.q.add(Dense(units=state_dim*3, activation='relu'))
        self.q.add(Dense(units=state_dim*3, activation='relu'))
        self.q.add(Dense(units=action_dim, activation='softmax'))
        self.q.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])


class QCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v1')

        state_dim = self.env.observation_space.shape[0] # Number of dims per layer
        action_dim = self.env.action_space.n
        self.q_network = QNetwork(state_dim, action_dim)

    def get_learning_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def get_explore_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def choose_action(self, state, explore_rate):
        return self.env.action_space.sample() if (np.random.random() < explore_rate) else 0

    # def update_q(self, state_0, action, reward, state, learning_rate):
        # self.q[state_0][action] += learning_rate*(reward + 0.99 * np.max(self.q[state]) - self.q[state_0][action])

    def update_network(self):
        pass

    def run(self):
        rewards = []
        for i in range(1000):
            state_0 = self.env.reset()

            learning_rate = self.get_learning_rate(i)
            explore_rate = self.get_explore_rate(i)
            episode_rewards = 0

            for j in range(500):
                # self.env.render()
                action = self.choose_action(state_0, explore_rate)
                obs, reward, done, _ = self.env.step(action)

                state = obs
                # self.update_q(state_0, action, reward, state, learning_rate)
                self.update_network()

                state_0 = state
                episode_rewards += reward

                if done:
                    print("Episode %d finished in %d timesteps." % (i, j))
                    rewards.append(episode_rewards)
                    break
        self.env.close()
        print(rewards[-100:])
        print(np.mean(rewards[-100:]))
        print(np.max(rewards[-100:]))

if __name__ == "__main__":
    solver = QCartPoleSolver()
    solver.run()
