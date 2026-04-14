import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import os

class DoubleDQNAgent_LunarLander:
    def __init__(self, env_name='LunarLander-v3'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)

        # Hiperparâmetros otimizados para o LunarLander
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_dec = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)

        self.value_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.value_network.get_weights())

    def experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.value_network.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Double DQN: value_network seleciona a melhor ação, target_network avalia o Q-value
        next_q_values_value = self.value_network.predict_on_batch(next_states)
        best_actions = np.argmax(next_q_values_value, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        indexes = np.arange(self.batch_size)
        next_q = next_q_values_target[indexes, best_actions]

        targets = rewards + self.gamma * next_q * (1 - dones)
        targets_full = self.value_network.predict_on_batch(states)
        targets_full[indexes, actions] = targets

        self.value_network.fit(states, targets_full, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec

    def train(self, episodes=200, max_steps=200, target_update_freq=10, target_update_per_step=False, label="DDQN"):
        rewards_history = []
        for e in range(episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0

            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                total_reward += reward
                is_terminal = done or truncated
                
                next_state_reshaped = np.reshape(next_state, [1, self.state_size])
                self.experience(state, action, reward, next_state_reshaped, is_terminal)
                
                state = next_state_reshaped
                self.experience_replay()

                if target_update_per_step:
                    self.update_target_network()
                
                if is_terminal:
                    print(f"{label} Ep: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")
                    break

            if not target_update_per_step and (e + 1) % target_update_freq == 0:
                self.update_target_network()

            rewards_history.append(total_reward)
        return rewards_history

    def save_weights(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.value_network.save_weights(filename)

    def evaluate_policy(self, n_episodes=100):
        env = gym.make(self.env_name)
        evaluation_rewards = []
        old_epsilon = self.epsilon
        self.epsilon = 0.0 # Inferência pura, zero exploração aleatória
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            obs = np.reshape(obs, [1, self.state_size])
            done = truncated = False
            ep_reward = 0
            
            while not (done or truncated):
                action = self.select_action(obs)
                next_obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                obs = np.reshape(next_obs, [1, self.state_size])
                
            evaluation_rewards.append(ep_reward)
            
        env.close()
        self.epsilon = old_epsilon
        return evaluation_rewards

    def generate_gif(self, filename, max_steps=1000):
        import imageio
        env = gym.make(self.env_name, render_mode="rgb_array")
        frames = []

        obs, _ = env.reset()
        obs = np.reshape(obs, [1, self.state_size])
        done = truncated = False

        for _ in range(max_steps):
            frames.append(env.render())
            action = np.argmax(self.value_network.predict(obs, verbose=0)[0])
            next_obs, reward, done, truncated, _ = env.step(action)
            obs = np.reshape(next_obs, [1, self.state_size])
            if done or truncated:
                frames.append(env.render())
                break

        env.close()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        imageio.mimsave(filename, frames, fps=30)