import gymnasium as gym
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import os

class DQNAgent_MountainCar:
    def __init__(self, env_name='MountainCar-v0'):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)

        # Hiperparâmetros otimizados para o MountainCar
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_dec = 0.995 # Decaimento mais lento para explorar mais
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=20000)

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,))) # Forma moderna do Keras 3
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
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

        next_max = np.amax(self.model.predict_on_batch(next_states), axis=1)
        targets = rewards + self.gamma * next_max * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        targets_full[[indexes], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec

    def train(self, episodes=200, max_steps=200):
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
                
                if is_terminal:
                    print(f"DQN Ep: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {self.epsilon:.2f}")
                    break
            rewards_history.append(total_reward)
        return rewards_history

    def save_weights(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.model.save_weights(filename)

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