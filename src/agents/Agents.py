import gymnasium as gym
import numpy as np
from tqdm import tqdm
import time
import math
import os

class AgentRL_Mountain_Car:
    def __init__(self, env_name='MountainCar-v0', algo="q_learning", alpha=0.1, epsilon=1.0, gamma=0.99):
        self.env_name = env_name
        self.algo = algo.lower() 
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make(env_name)

        # Discretizando o espaco de estados (Exatamente como o professor pediu)
        self.num_states = (self.env.observation_space.high - self.env.observation_space.low)*np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1

        # Inicializando uma q-table com 3 dimensoes: posicao, velocidade, acao
        self.Q = np.zeros([self.num_states[0], self.num_states[1], self.env.action_space.n])
    
    def discretize_state(self, state):
        """Converte o estado contínuo do Mountain Car em índices discretos."""
        scaling_factor = np.array([10, 100])
        state_adj = (state - self.env.observation_space.low) * scaling_factor
        state_adj = np.round(state_adj, 0).astype(int)
        
        # Segurança contra bounds
        state_adj[0] = min(self.num_states[0] - 1, max(0, state_adj[0]))
        state_adj[1] = min(self.num_states[1] - 1, max(0, state_adj[1]))
        
        return tuple(state_adj)
    
    def arg_max(self, q_values):
        ties = []
        top_value = float('-inf')
        for i in range(len(q_values)):
            if q_values[i] > top_value:
                ties = []
                top_value = q_values[i]
                ties.append(i)
            elif q_values[i] == top_value:
                ties.append(i)
        return np.random.choice(ties)
    
    def greedy_selection(self, Q_state):
        return self.arg_max(Q_state)
    
    def epsilon_greedy_selection(self, Q_state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(Q_state))
        else:
            return self.arg_max(Q_state)
        
    def incremental_update(self, Q, state, action, reward, next_state, next_action, done):
        if done:
            target = reward
        else:
            if self.algo == "sarsa":
                target = reward + self.gamma * Q[next_state][next_action]
            elif self.algo == "q_learning":
                target = reward + self.gamma * np.max(Q[next_state])
            else:
                raise ValueError("Algoritmo não reconhecido. Use 'sarsa' ou 'q_learning'.")
        
        Q[state][action] = Q[state][action] + self.alpha * (target - Q[state][action])
        return Q
    
    def run_algorithm(self, episodes=5000, max_steps=200):
        Q = self.Q
        rewards_per_episode = np.zeros(episodes)
        alphas = [] 
        epsilons = [] 

        epsilon_inicial = 1.0
        epsilon_final = 0.01
        alpha_inicial = self.alpha
        alpha_final = 0.01

        # Decaimento linear
        decay_rate = (epsilon_inicial - epsilon_final) / (0.8 * episodes)
        decay_rate_alpha = (alpha_inicial - alpha_final) / (0.8 * episodes)

        for ep in tqdm(range(episodes), desc=f"Training in {self.env_name} ({self.algo.upper()})"):
            
            self.epsilon = max(epsilon_final, epsilon_inicial - (ep * decay_rate))

            obs_continuous, info = self.env.reset()
            obs = self.discretize_state(obs_continuous)
            
            done = False
            truncated = False
            step_count = 0
            total_reward = 0  # CORREÇÃO 1: Acumulador de recompensa do episódio
            
            action = self.epsilon_greedy_selection(Q[obs])

            while not (done or truncated) and step_count < max_steps:
                next_obs_continuous, step_reward, done, truncated, info = self.env.step(action)
                
                total_reward += step_reward # Acumulando a recompensa (-1 por passo)

                next_obs = self.discretize_state(next_obs_continuous)
                next_action = self.epsilon_greedy_selection(Q[next_obs])

                Q = self.incremental_update(Q, obs, action, step_reward, next_obs, next_action, done)

                obs = next_obs
                action = next_action
                step_count += 1

            # Salvando a recompensa total real do episódio no array
            rewards_per_episode[ep] = total_reward

            self.alpha = max(0.001, alpha_inicial - (ep * decay_rate_alpha))
            alphas.append(self.alpha)
            epsilons.append(self.epsilon)

        self.env.close()
        return rewards_per_episode, Q, alphas, epsilons

    def save_weights(self, filename):
        """CORREÇÃO 2: Salva a Q-Table em um arquivo numpy para entregar na atividade"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, self.Q)
        
    def load_weights(self, filename):
        """Carrega a Q-Table salva"""
        self.Q = np.load(filename)

    def evaluate_policy(self, n_episodes=100):
        """
        CORREÇÃO 3: Retorna um array com as recompensas de cada episódio
        Isso é necessário para plotar o gráfico de inferência
        """
        env = gym.make(self.env_name) 
        evaluation_rewards = []
        
        # Desliga a exploração para inferência pura
        old_epsilon = self.epsilon
        self.epsilon = 0.0 
        
        for _ in range(n_episodes):
            obs_continuous, _ = env.reset()
            obs = self.discretize_state(obs_continuous)
            done = truncated = False
            ep_reward = 0
            
            while not (done or truncated):
                action = self.greedy_selection(self.Q[obs])
                next_obs_continuous, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                obs = self.discretize_state(next_obs_continuous)
                
            evaluation_rewards.append(ep_reward)
            
        env.close()
        self.epsilon = old_epsilon # Restaura o epsilon
        
        return evaluation_rewards

    def view(self, Q, episodes=3, max_steps=100, sleep=0.3):
        env = gym.make(self.env_name, render_mode="human", is_slippery=True)

        for ep in range(episodes):
            obs, info = env.reset()
            total = 0 
            print(f"Iniciando Episódio {ep+1}")

            for s in range(max_steps):
                # We need to call render() to ensure the window updates in some backends
                env.render() 
                
                action = self.greedy_selection(Q[obs])
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                time.sleep(sleep)

                if terminated or truncated:
                    # Final render to show the result (Goal or Hole)
                    env.render()
                    time.sleep(1)
                    break
            
            print(f"[{self.algo.upper()}] Fim do Episódio {ep+1}: Retorno = {total}")


        env.close()

    def get_optimal_value(self, gamma=0.99, theta=1e-4):
        """
        Calcula o Value Iteration para o Mountain Car discretizado.
        Como não existe env.P, simulamos a física do ambiente.
        """
        # A matriz V agora tem 2 dimensões (posição e velocidade)
        V = np.zeros((self.num_states[0], self.num_states[1]))
        n_actions = self.env.action_space.n

        # Usado para reverter o índice discreto para o valor contínuo
        low = self.env.observation_space.low
        scaling_factor = np.array([10, 100])

        print("Calculando Value Iteration (isso pode levar alguns segundos)...")
        
        # Value Iteration
        while True:
            delta = 0
            
            # Percorre todas as posições e velocidades da nossa grade discretizada
            for p in range(self.num_states[0]):
                for v in range(self.num_states[1]):
                    v_old = V[p, v]
                    
                    # Converte os índices da matriz de volta para valores contínuos reais
                    cont_pos = (p / scaling_factor[0]) + low[0]
                    cont_vel = (v / scaling_factor[1]) + low[1]

                    # Estado Terminal: Se já passou da bandeira, o valor é 0 e encerra
                    if cont_pos >= 0.5:
                        V[p, v] = 0.0
                        continue

                    q_values = []
                    for action in range(n_actions):
                        # ---------------------------------------------------------
                        # SIMULANDO A FÍSICA DO MOUNTAIN CAR
                        # Calculamos a nova velocidade considerando aceleração (ação) e gravidade (cosseno)
                        next_vel = cont_vel + (action - 1) * 0.001 + math.cos(3 * cont_pos) * (-0.0025)
                        next_vel = np.clip(next_vel, -0.07, 0.07)
                        
                        # Calculamos a nova posição
                        next_pos = cont_pos + next_vel
                        next_pos = np.clip(next_pos, -1.2, 0.6)
                        
                        # Colisão com a parede esquerda (zera a velocidade)
                        if next_pos == -1.2 and next_vel < 0:
                            next_vel = 0.0
                        # ---------------------------------------------------------

                        # Discretiza o resultado físico para achar o índice do próximo estado na matriz
                        next_state_cont = np.array([next_pos, next_vel])
                        next_p, next_v = self.discretize_state(next_state_cont)

                        # A recompensa no Mountain Car é sempre -1 por passo
                        reward = -1.0
                        
                        # Equação de Bellman
                        q_sa = reward + gamma * V[next_p, next_v]
                        q_values.append(q_sa)
                    
                    # Atualiza o valor do estado com a melhor ação possível
                    V[p, v] = max(q_values)
                    delta = max(delta, abs(v_old - V[p, v]))
            
            if delta < theta:
                break
                
        print("Value Iteration Concluído!")
        return V # Retorna a matriz inteira de valores
