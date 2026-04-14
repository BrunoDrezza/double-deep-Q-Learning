import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Estilo bonito pros gráficos
sns.set_theme(style="darkgrid")

def plot_learning_curves(q_rewards, dqn_rewards, filename):
    plt.figure(figsize=(10, 6))
    
    # Prepara dados do Q-Learning
    df_q = pd.DataFrame(q_rewards).melt(var_name='Episódio', value_name='Recompensa')
    df_q['Algoritmo'] = 'Q-Learning'
    
    # Prepara dados do DQN
    df_dqn = pd.DataFrame(dqn_rewards).melt(var_name='Episódio', value_name='Recompensa')
    df_dqn['Algoritmo'] = 'Deep Q-Learning'
    
    # Junta tudo e plota
    df_all = pd.concat([df_q, df_dqn])
    sns.lineplot(data=df_all, x='Episódio', y='Recompensa', hue='Algoritmo')
    
    # Linha da Meta do Mountain Car
    plt.axhline(y=-110, color='red', linestyle='--', linewidth=2, label='Meta (-110)')
    
    plt.title('Curva de Aprendizado: Q-Learning vs Deep Q-Learning (5 Execuções)')
    plt.ylabel('Recompensa Acumulada')
    plt.xlabel('Episódios')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_inference_comparison(q_inf_rewards, dqn_inf_rewards, filename):
    plt.figure(figsize=(8, 5))
    algoritmos = ['Q-Learning', 'Deep Q-Learning']
    medias = [np.mean(q_inf_rewards), np.mean(dqn_inf_rewards)]
    
    sns.barplot(x=algoritmos, y=medias, palette="viridis")
    plt.axhline(y=-110, color='red', linestyle='--', linewidth=2, label='Meta (-110)')
    
    plt.title('Desempenho dos Agentes na Inferência (Sem Exploração)')
    plt.ylabel('Recompensa Média')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()