import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import load_phishing_dataset

class PhishingDatasetPreproc():

    def init(self):
        self.dataset = load_phishing_dataset()
        self.target = 'Result'

    def basic_operations(self):
        '''
        Basic preprocessing operations:
        - Converting target variable values from {-1,1} to {0,1} for use in some models.
        - Splitting features and target variable.
        '''
        trg = self.target

        # Converting from {-1,1} to {0,1} for use in some models
        self.dataset[trg] = self.dataset[trg].astype(int)
        self.dataset.loc[self.dataset[trg] == -1, trg] = 0

        # Splitting between features and target variable
        self.X = self.dataset.drop(columns=trg)
        self.y = self.dataset[trg]

    def tree_based_feature_selection(dataset, target = 'Result'):
        '''
        Simple tree-based feature selection.
        '''
        dataset[target] = dataset['Result'].astype(int)

        model = RandomForestClassifier()
        model.fit(dataset.drop(target, axis=1), dataset[target])
        
        feature_importances = pd.Series(model.feature_importances_, index=dataset.drop(target, axis=1).columns)
        selected_features = feature_importances[feature_importances > 0.01].index

        return selected_features

