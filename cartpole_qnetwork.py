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
        # self.q.add(Dense(units=state_dim*3, activation='relu'))
        self.q.add(Dense(units=action_dim, activation=None))
        self.q.compile(loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])

    def predict(self, state):
        return self.q.predict(state)

    def update(self, target, curr_state):
        return self.q.fit(curr_state, target)

class QCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.discount_rate = 0.99

        state_dim = self.env.observation_space.shape[0] # Number of dims per layer
        action_dim = self.env.action_space.n
        self.q_network = QNetwork(state_dim, action_dim)

    def get_learning_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def get_explore_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def choose_action(self, state, explore_rate):
        state = state.reshape(1, -1)
        return self.env.action_space.sample() if (np.random.random() < explore_rate) else np.argmax(self.q_network.predict(state))

    def update_network(self, curr_state, action, reward, next_state):
        next_state = next_state.reshape(1, -1)
        curr_state = curr_state.reshape(1, -1)
        new_q = reward + self.discount_rate * np.max(self.q_network.predict(next_state))
        target = self.q_network.predict(curr_state)
        print(target)
        target[0][action] = new_q
        self.q_network.update(target, curr_state)

    def run(self):
        rewards = []
        for i in range(1000):
            curr_state = self.env.reset()

            learning_rate = self.get_learning_rate(i)
            explore_rate = self.get_explore_rate(i)
            episode_rewards = 0

            for j in range(500):
                self.env.render()
                action = self.choose_action(curr_state, explore_rate)
                obs, reward, done, _ = self.env.step(action)

                next_state = obs
                self.update_network(curr_state, action, reward, next_state)

                curr_state = next_state
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
