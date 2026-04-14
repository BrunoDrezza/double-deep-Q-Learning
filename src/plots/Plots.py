import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Estilo bonito pros gráficos
sns.set_theme(style="darkgrid")

def plot_learning_curves(rewards_dict, filename):
    """
    rewards_dict: dict com label -> lista de recompensas por episódio.
    """
    plt.figure(figsize=(12, 6))

    frames = []
    for label, rewards in rewards_dict.items():
        df = pd.DataFrame({'Episódio': range(len(rewards)), 'Recompensa': rewards})
        df['Cenário'] = label
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    sns.lineplot(data=df_all, x='Episódio', y='Recompensa', hue='Cenário')

    plt.axhline(y=200, color='red', linestyle='--', linewidth=2, label='Meta (200)')

    plt.title('Curva de Aprendizado – LunarLander-v3')
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