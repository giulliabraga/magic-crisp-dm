import random
import numpy as np
import optuna
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from model_selector import MODEL_MAP, PARAM_DICT_MAP

def optimize_pipeline(X, y, model_name, n_trials=50, cv_folds=5, scoring='accuracy'):
    """
    Optimize model hyperparameters using Optuna.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    - model_name (str): Model to evaluate (e.g., ["KNN"]).
    - n_trials (int): Number of optimization trials.
    - cv_folds (int): Number of cross-validation folds.
    - scoring (str): Scoring metric for cross-validation.

    Returns:
    - study.best_trial: The best trial from the optimization study.
    """
    def objective(trial):
        # Get the model and its hyperparameter search space
        model, param_dict = MODEL_MAP[model_name], PARAM_DICT_MAP[model_name]

        # Sample model hyperparameters dynamically
        model_params = {}
        for param_name, param_values in param_dict.items():

            if isinstance(param_values[0], int):
                # Sampling ints
                model_params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))

            elif isinstance(param_values[0], float):
                # Sampling floats
                if (max(param_values) / min(param_values)) >= 10:
                    # If the range is too big, use the log domain
                    log_flag = True
                else:
                    log_flag = False

                model_params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values), log=log_flag)

            elif isinstance(param_values[0], str):
                # Sampling categorical hyperparameters
                model_params[param_name] = trial.suggest_categorical(param_name, param_values)

            else:
                # Specifically for the MLP hidden_layer_sizes
                par = trial.suggest_categorical(param_name, [str(tup) for tup in param_values])
                model_params[param_name] = eval(par)

        model.set_params(**model_params)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score='raise')
        except ValueError as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        
        return scores.mean()
    
    study = optuna.create_study(study_name=f'optimization_{model_name}',direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # Print the best trial
    print("Best trial:")
    print(f"Value: {study.best_trial.value}")
    print("Params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    return study.best_trial

