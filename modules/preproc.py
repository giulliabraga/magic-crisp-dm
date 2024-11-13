import pandas as pd
from utils import load_phishing_dataset
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn import datasets

class PhishingDatasetPreproc():

    def __init__(self):
        self.dataset = load_phishing_dataset()
        self.target = 'Result'

    def basic_operations(self):
        '''
        Basic preprocessing operations:
        - Converting all data to int, as some models cannot deal with categorical data :(
        - Converting target variable values from {-1,1} to {0,1} for use in some models
        - Splitting features and target variable
        '''
        trg = self.target
        
        # Converting the entire dataset to int
        self.dataset = self.dataset.astype(int)

        # Converting from {-1,1} to {0,1} for use in some models
        self.dataset.loc[self.dataset[trg] == -1, trg] = 0

        # Splitting between features and target variable
        self.X = self.dataset.drop(columns=trg)
        self.y = self.dataset[trg]

        return self.dataset, self.X, self.y

def feature_selection_pipeline(model, param_grid=dict, fs_method=str, X=pd.DataFrame, y=pd.Series, n_iter=10):
    '''
    Creates and runs a pipeline with a specified feature selection method and model.
    
    model: model to use in the pipeline (e.g., LogisticRegression())
    param_grid: dictionary with hyperparameters for both the feature selection and model
    fs_method: str specifying feature selection ("tree", "pca", "univariate", "l1")
    n_iter: number of random samples for RandomizedSearchCV
    '''
    fs_methods = {
        "tree": SelectFromModel(estimator=RandomForestClassifier(n_estimators=100)),
        "pca": PCA(),
        "univariate": SelectKBest(f_classif),
        "l1": SelectFromModel(estimator=Lasso(alpha=0.1))
    }
    
    if fs_method in fs_methods.keys():
        print(fs_method)
        feature_selector = fs_methods[fs_method]
        print(feature_selector)
    else:
        raise ValueError("Unknown feature selection method")
        
    pipeline = Pipeline([
        ("selector", feature_selector), 
        ("model", model)
    ])
    
    search = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, n_jobs=2, random_state=42, verbose=1)

    search.fit(X,y)
    
    # Return best score and params
    return search.best_score_, search.best_params_

def eval_feature_selectors(model, param_grids, X, y, n_iter=10):
    '''
    Evaluates multiple feature selection methods and returns a DataFrame with the results.
    
    model: model to use in the pipeline (e.g., LogisticRegression())
    param_grids: list of dictionaries with hyperparameters for each feature selection and model
    X: DataFrame with features
    y: Series or array with target variable
    n_iter: number of random samples for RandomizedSearchCV
    '''
    selectors = ["tree", "pca", "univariate", "l1"]
    results = []

    for selector, param_grid in zip(selectors, param_grids):
        best_score, best_params = feature_selection_pipeline(
            model=model,
            param_grid=param_grid,
            fs_method=selector,
            X=X,
            y=y,
            n_iter=n_iter
        )
        results.append({"selector_name": selector, "best_cv_score": best_score, "best_params": best_params})

    return pd.DataFrame(results)
        
    

