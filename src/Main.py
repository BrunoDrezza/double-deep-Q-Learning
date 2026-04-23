import os
import sys

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_BACKEND'] = "torch"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow import keras
from src.agents.Agent_NN import DoubleDQNAgent_LunarLander
from src.plots.Plots import plot_learning_curves
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    plots_dir = os.path.join(base_dir, "plots", "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    EPISODES = 100
    MAX_STEPS = 1000
    all_rewards = {}

    # ======================================================
    # Cenário 1: DQN Padrão (target update a cada step)
    # ======================================================
    print("=" * 60)
    print("CENÁRIO 1: DQN Padrão (target update a cada step)")
    print("=" * 60)
    agent1 = DoubleDQNAgent_LunarLander()
    rewards1 = agent1.train(episodes=EPISODES, max_steps=MAX_STEPS,
                            target_update_per_step=True, label="DQN")
    all_rewards["DQN_standard"] = rewards1
    agent1.save_weights(os.path.join(data_dir, "dqn_standard_weights.h5"))
    del agent1
    keras.backend.clear_session()
    print("Cenário 1 concluído.\n")

    # ======================================================
    # Cenário 2: Double DQN com C = 1
    # ======================================================
    print("=" * 60)
    print("CENÁRIO 2: Double DQN com C = 1")
    print("=" * 60)
    agent2 = DoubleDQNAgent_LunarLander()
    rewards2 = agent2.train(episodes=EPISODES, max_steps=MAX_STEPS,
                            target_update_freq=1, label="DDQN C=1")
    all_rewards["DDQN_C1"] = rewards2
    agent2.save_weights(os.path.join(data_dir, "ddqn_c1_weights.h5"))
    del agent2
    keras.backend.clear_session()
    print("Cenário 2 concluído.\n")

    # ======================================================
    # Cenário 3: Double DQN com C = 10 (+ GIFs)
    # ======================================================
    print("=" * 60)
    print("CENÁRIO 3: Double DQN com C = 10 (cenário principal)")
    print("=" * 60)
    agent3 = DoubleDQNAgent_LunarLander()

    # Treina primeiros 100 episódios → agente ruim
    rewards3_part1 = agent3.train(episodes=100, max_steps=MAX_STEPS,
                                  target_update_freq=10, label="DDQN C=10")
    agent3.generate_gif(os.path.join(plots_dir, "agente_ruim.gif"))
    print(">>> agente_ruim.gif gerado (episódio 100)")

    # Treina os 900 episódios restantes → agente bom
    rewards3_part2 = agent3.train(episodes=900, max_steps=MAX_STEPS,
                                  target_update_freq=10, label="DDQN C=10")
    all_rewards["DDQN_C10"] = rewards3_part1 + rewards3_part2
    agent3.save_weights(os.path.join(data_dir, "ddqn_c10_weights.h5"))
    agent3.generate_gif(os.path.join(plots_dir, "agente_bom.gif"))
    print(">>> agente_bom.gif gerado (episódio 1000)")
    del agent3
    keras.backend.clear_session()
    print("Cenário 3 concluído.\n")

    # ======================================================
    # Cenário 4: Double DQN com C = 50
    # ======================================================
    print("=" * 60)
    print("CENÁRIO 4: Double DQN com C = 50")
    print("=" * 60)
    agent4 = DoubleDQNAgent_LunarLander()
    rewards4 = agent4.train(episodes=EPISODES, max_steps=MAX_STEPS,
                            target_update_freq=50, label="DDQN C=50")
    all_rewards["DDQN_C50"] = rewards4
    agent4.save_weights(os.path.join(data_dir, "ddqn_c50_weights.h5"))
    del agent4
    keras.backend.clear_session()
    print("Cenário 4 concluído.\n")

    # Salva os históricos de recompensas para uso posterior nos plots
    for name, rewards in all_rewards.items():
        np.save(os.path.join(data_dir, f"rewards_{name}.npy"), rewards)
    print("Históricos de recompensas salvos em /data/")

    # Gera gráfico comparativo dos 4 cenários
    plot_rewards = {
        "DQN Padrão": all_rewards["DQN_standard"],
        "DDQN C=1": all_rewards["DDQN_C1"],
        "DDQN C=10": all_rewards["DDQN_C10"],
        "DDQN C=50": all_rewards["DDQN_C50"],
    }
    plot_learning_curves(plot_rewards, os.path.join(plots_dir, "lunarlander_comparison.png"))
    print("Gráfico comparativo salvo em plots/results/lunarlander_comparison.png")

if __name__ == "__main__":
    main()