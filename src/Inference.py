import os
import sys
import glob
import gymnasium as gym
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.Agent_NN import DoubleDQNAgent_LunarLander


def find_best_weights(data_dir="data"):
    pattern = os.path.join(data_dir, "*C10*.h5")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Fallback: qualquer .h5 disponível
    all_h5 = glob.glob(os.path.join(data_dir, "*.h5"))
    if all_h5:
        return all_h5[0]
    return None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    weights_file = find_best_weights(data_dir)
    if weights_file is None:
        print("Nenhum arquivo .h5 encontrado em data/. Treine o agente primeiro.")
        return

    print(f"Carregando pesos de: {weights_file}")
    agent = DoubleDQNAgent_LunarLander()
    agent.load_weights(weights_file)

    env = gym.make("LunarLander-v3", render_mode="human")

    for ep in range(10):
        obs, _ = env.reset()
        obs = np.reshape(obs, [1, agent.state_size])
        done = truncated = False
        total_reward = 0

        while not (done or truncated):
            action = np.argmax(agent.value_network.predict(obs, verbose=0)[0])
            next_obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            obs = np.reshape(next_obs, [1, agent.state_size])

        print(f"Episódio {ep + 1}/10 — Recompensa: {total_reward:.2f}")

    env.close()
    print("Inferência concluída.")


if __name__ == "__main__":
    main()
