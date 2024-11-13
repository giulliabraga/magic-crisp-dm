from matplotlib import pyplot as plt
import pandas as pd
from os import X_OK
from sklearn.decomposition import PCA
import seaborn as sns
import umap


def load_phishing_dataset(filepath = '../data/phishing-dataset.arff'):
    '''
    Load the phishing websites arff file
    '''

    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = [line.strip() for line in lines if not line.startswith('@') and line.strip()]

    data = [line.split(',') for line in data_lines]
    col_names = [line.split()[1] for line in lines if line.startswith('@attribute')]
    dataset = pd.DataFrame(data, columns=col_names)

    return dataset

def quick_stats(dataset):
    '''
    Runs a quick statistical summary of the dataset
    '''
    stats = dataset.describe().T

    # Unique value counts
    value_counts = dataset.apply(lambda col: col.value_counts()).T

    # Output containin the stats and the counts for each unique value
    stats = (
                pd.concat([stats,value_counts],axis=1)
                .fillna(0)
                .style
                .format(precision=0)
            )

    return stats

def plot_pca(dataset, target='Result'):
    '''
    Generates a 2D visualization of a dataset using PCA.
    '''
    pca = PCA(2)

    X = dataset.drop(columns=target)
    y = dataset[target]

    pca_result = pca.fit_transform(X)

    dataset_pca = pd.DataFrame({
        'X': pca_result[:, 0],
        'y': pca_result[:, 1],
        'target': y
    })

    fig = plt.figure(figsize=(22, 8))

    sns.scatterplot(data=dataset_pca, x='X', y='y', hue='target', palette='viridis', s=100, edgecolor='k')

    plt.title('PCA', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=16)
    plt.ylabel('Principal Component 2', fontsize=16)
    plt.legend(title=target, title_fontsize='13', fontsize='11')

    plt.show()

def plot_umap(dataset, target='Result'):
    '''    
    Generates a 2D visualization of a dataset using UMAP (supervised).
    '''
    X = dataset.drop(columns=target)
    y = dataset[target]

    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(X,y=y)

    dataset_umap = pd.DataFrame({
        'X': umap_result[:, 0],
        'y': umap_result[:, 1],
        'target': y
    })


    fig = plt.figure(figsize=(22, 8))

    sns.scatterplot(data=dataset_umap, x='X', y='y', hue='target', palette='viridis', s=100, edgecolor='k')

    plt.title('UMAP', fontsize=20)
    plt.xlabel('Dimension 1', fontsize=16)
    plt.ylabel('Dimension 2', fontsize=16)
    plt.legend(title=target, title_fontsize='13', fontsize='11')

    plt.show()