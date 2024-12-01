import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

def plot_metric_boxplots(list_of_model_results, list_of_model_names, list_of_metrics_names):
    """
    Gera uma figura com subplots organizados em duas colunas, onde cada subplot contém boxplots
    para uma métrica específica de todos os modelos fornecidos.

    Parameters:
        list_of_model_results (list): Lista de DataFrames, cada um contendo os resultados de um modelo.
        list_of_model_names (list): Lista com os nomes dos modelos correspondentes.
        list_of_metrics_names (list): Lista com os nomes das métricas a serem plotadas.

    Returns:
        None: Exibe o gráfico com os boxplots.
    """
    # Preparando os dados para o plot
    combined_data = []
    for model_name, model_results in zip(list_of_model_names, list_of_model_results):
        df = pd.DataFrame(model_results)
        df['model'] = model_name
        combined_data.append(df)
    
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Número de métricas e configuração do layout
    n_metrics = len(list_of_metrics_names)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)

    # Criando a figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows), sharey=False)
    axes = axes.flatten()  # Para fácil indexação

    for i, metric in enumerate(list_of_metrics_names):
        ax = axes[i]
        if metric not in combined_df.columns:
            raise ValueError(f"Métrica '{metric}' não encontrada nos resultados.")
        
        # Criando os boxplots
        combined_df.boxplot(
            column=metric,
            by='model',
            ax=ax,
            grid=False,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red'),
        )
        ax.set_title(metric)
        ax.set_xlabel('Modelos')
        ax.set_ylabel(metric)
    
    # Remove eixos vazios se houver menos métricas que subplots disponíveis
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajusta layout e remove o título automático
    plt.tight_layout()
    fig.suptitle("")  # Remove título automático do pandas
    plt.show()

def plot_time_vs_performance_scatter(list_of_model_results, list_of_model_names, list_of_metrics_names):
    # Unir os dados em um único DataFrame
    combined_data = []
    for model_name, model_results in zip(list_of_model_names, list_of_model_results):
        df = pd.DataFrame(model_results)
        df['model'] = model_name
        combined_data.append(df)
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Número de métricas e configuração de layout
    n_metrics = len(list_of_metrics_names)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)

    # Criar a figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows))
    axes = axes.flatten()

    # Criar uma paleta de cores pastéis diferenciadas
    palette = sns.color_palette("pastel", len(list_of_model_names))  # Tons pastéis

    for i, metric in enumerate(list_of_metrics_names):
        ax = axes[i]
        if metric not in combined_df.columns:
            raise ValueError(f"Métrica '{metric}' não encontrada.")

        # Scatter plot com a paleta pastel
        scatter = sns.scatterplot(
            data=combined_df,
            x="training_time",
            y="inference_time",
            hue="model",  # Modelos diferenciados por cor
            size=metric,  # Métrica no tamanho
            sizes=(20, 200),
            palette=palette,  # Aplicando a paleta pastel
            ax=ax,
        )
        ax.set_title(f"{metric} vs Tempo", fontsize=14)
        ax.set_xlabel("Tempo de Treinamento (s)", fontsize=12)
        ax.set_ylabel("Tempo de Inferência (s)", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

    # Remover subplots extras
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout e legenda
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Modelos")
    plt.show()