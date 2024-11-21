from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lvq_classifier import LVQ

'''
This module allows for an easy selection of a model with its respective parameter grid for optimization.
'''

model_dicts = [
                    {   
                        "model": "KNN",
                        "params_dict": {
                            "n_neighbors": [3,4,5,6,7,8,9,10],
                            "weights":["uniform", "distance"],
                            "algorithm":["auto", "ball_tree","kd_tree","brute"],
                            "metric" : ["euclidean", "manhattan", "chebyshev"]
                        },
                        "import": KNeighborsClassifier()
                    },
                    {
                        "model": "LVQ",
                        "params_dict":{
                            'n_codebooks': [5, 10],
                            'lrate': [0.1, 0.01],
                            'epochs': [20, 50, 100]
                        },
                        "import": LVQ()
                    }, 
                    {
                        "model": "DTR",
                        "params_dict":{
                            'criterion':['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [3, 5, 10],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [2, 3, 4, 5],
                            'max_features': ['sqrt', 'log2'],
                        },
                        "import": DecisionTreeClassifier()
                    },
                    {
                        "model": "SVM",
                        "params_dict": {
                            'C': [0.1,1,5,10,20,50,100],
                            'kernel': ['linear','poly','rbf','sigmoid'],
                            'gamma': [1,5,10,50,100,500],
                        },
                        "import": SVC(probability=True)
                    },
                    {
                        "model": "RF",
                        "params_dict": {
                            'n_estimators': [10, 50, 100, 200, 500],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [3, 5, 10],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [2, 3, 4, 5],
                            'max_features': ['auto', 'sqrt', 'log2'],
                        },
                        "import": RandomForestClassifier(),
                    },
                    {
                        "model": "XGB",
                        "params_dict": {
                            'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                            'loss': ['log_loss', 'exponential'],
                            'n_estimators': [100, 200, 300, 400, 500, 700, 1000],
                            'max_depth': [3, 4, 5],
                            'subsample': [0.3, 0.5, 0.7, 0.9],
                            'criterion': ['friedman_mse', 'squared_error']
                        },
                        "import": GradientBoostingClassifier(),
                    },
                    {
                        "model": "LGBM",
                        "params_dict": {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'num_leaves': [31, 50, 70],
                            'max_depth': [-1, 10, 20],
                            'subsample': [0.8, 1.0]
                        },
                        "import": LGBMClassifier()
                    },
                    {
                        "model": "MLP",
                        "params_dict": {
                            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['adam', 'sgd'],
                            'alpha': [0.0001, 0.001, 0.01]
                        },
                        "import": MLPClassifier()
                    }
                ]

# Mapping parameter dictionaries
PARAM_DICT_MAP = {
    param["model"]: param["params_dict"] for param in model_dicts
}

MODEL_MAP = {
    param["model"]: param["import"] for param in model_dicts
}

def get_model_and_params(model_name):
    """
    Retrieve a model instance and its parameter dictionary for RandomizedSearchCV.

    Parameters:
    - model_name (str): The name of the model, as a string. Must be one of the keys in MODEL_MAP.

    Returns:
    - model (estimator): An instance of the requested model.
    - param_dict (dict): The parameter dictionary for RandomizedSearchCV.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model '{model_name}' is not recognized. Available models are: {list(MODEL_MAP.keys())}")

    if model_name not in PARAM_DICT_MAP:
        raise ValueError(f"Model '{model_name}' does not have a parameter dictionary. Available models are: {list(PARAM_DICT_MAP.keys())}")

    model = MODEL_MAP[model_name]()
    param_dict = PARAM_DICT_MAP[model_name]
    
    return model, param_dict