import gym
import math
import numpy as np

# Referenced https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947 
# Referenced https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578

class QCartPoleSolver():
    def __init__(self):
        self.env = gym.make('CartPole-v1')

        self.buckets = (1, 1, 6, 3)
        self.q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def get_learning_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def get_explore_rate(self, t):
        return max(0.1, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def choose_action(self, state, explore_rate):
        return self.env.action_space.sample() if (np.random.random() < explore_rate) else np.argmax(self.q[state])

    def update_q(self, state_0, action, reward, state, learning_rate):
        self.q[state_0][action] += learning_rate*(reward + 0.99 * np.max(self.q[state]) - self.q[state_0][action])

    def run(self):
        rewards = []
        for i in range(1000):
            state_0 = self.discretize(self.env.reset())

            learning_rate = self.get_learning_rate(i)
            explore_rate = self.get_explore_rate(i)
            episode_rewards = 0

            for j in range(500):
                self.env.render()
                action = self.choose_action(state_0, explore_rate)
                obs, reward, done, _ = self.env.step(action)

                state = self.discretize(obs)
                self.update_q(state_0, action, reward, state, learning_rate)

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