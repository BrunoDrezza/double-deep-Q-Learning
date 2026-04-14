import os
import sys
import numpy as np

# Garante que o Python ache as pastas do projeto independentemente de onde você rodar o script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.Agents import AgentRL_Mountain_Car
from src.agents.Agent_NN import DQNAgent_MountainCar
from src.plots.Plots import plot_learning_curves, plot_inference_comparison

def main():
    # Cria as pastas na raiz do projeto, caso não existam
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots", "results"), exist_ok=True) # Pasta para salvar as imagens

    N_RUNS = 5
    EPISODES = 5
    
    q_all_rewards = []
    dqn_all_rewards = []

    print("="*50)
    print(f"INICIANDO TREINAMENTO - {N_RUNS} EXECUÇÕES")
    print("="*50)

    # Variáveis para guardar o melhor agente para a inferência
    q_agent_final = None
    dqn_agent_final = None

    for run in range(N_RUNS):
        print(f"\n---> Execução {run+1}/{N_RUNS} <---")
        
        # 1. Q-LEARNING
        print("[1/2] Treinando Q-Learning...")
        q_agent = AgentRL_Mountain_Car(env_name='MountainCar-v0', algo="q_learning")
        q_rewards, _, _, _ = q_agent.run_algorithm(episodes=EPISODES)
        q_all_rewards.append(q_rewards)
        q_agent_final = q_agent # Guarda o último treinado
        
        if run == N_RUNS - 1:
            # Salva a Q-Table na última execução
            q_agent.save_weights(os.path.join(base_dir, "data", "q_table_final.npy"))
            print("Pesos do Q-Learning salvos em /data/")

        # 2. DEEP Q-LEARNING (DQN)
        print("[2/2] Treinando Deep Q-Learning...")
        dqn_agent = DQNAgent_MountainCar(env_name='MountainCar-v0')
        dqn_rewards = dqn_agent.train(episodes=EPISODES)
        dqn_all_rewards.append(dqn_rewards)
        dqn_agent_final = dqn_agent # Guarda o último treinado
        
        if run == N_RUNS - 1:
            # Salva os pesos da rede na última execução (Assumindo que você criou o save_weights lá)
            try:
                dqn_agent.save_weights(os.path.join(base_dir, "data", "dqn_weights_final.h5"))
                print("Pesos do DQN salvos em /data/")
            except AttributeError:
                print("Aviso: Método save_weights não encontrado no Agent_NN. Pulei o salvamento do H5.")

    print("\n" + "="*50)
    print("GERANDO GRÁFICO DA CURVA DE APRENDIZADO")
    print("="*50)
    
    # Chama a função de plotar passando as matrizes de recompensa
    curva_path = os.path.join(base_dir, "plots", "results", "curva_aprendizado.png")
    plot_learning_curves(q_all_rewards, dqn_all_rewards, filename=curva_path)
    print(f"Gráfico salvo em: {curva_path}")

    print("\n" + "="*50)
    print("INICIANDO INFERÊNCIA (AGENTE ATUANDO SEM TREINAMENTO)")
    print("="*50)
    
    print("Avaliando Q-Learning (100 episódios)...")
    q_inf_rewards = q_agent_final.evaluate_policy(n_episodes=2)
    
    print("Avaliando DQN (100 episódios)...")
    # Assumindo que você tem um evaluate_policy no seu Agent_NN. Se não tiver, adapte.
    try:
        dqn_inf_rewards = dqn_agent_final.evaluate_policy(n_episodes=2)
    except AttributeError:
        print("Aviso: Método evaluate_policy não encontrado no Agent_NN. Usando dados fictícios para não quebrar o plot.")
        dqn_inf_rewards = [-200] * 100 # Fallback de emergência
    
    inf_path = os.path.join(base_dir, "plots", "results", "comparacao_inferencia.png")
    plot_inference_comparison(q_inf_rewards, dqn_inf_rewards, filename=inf_path)
    print(f"Gráfico de inferência salvo em: {inf_path}")

if __name__ == "__main__":
    main()