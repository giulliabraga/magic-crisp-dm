import pandas as pd


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