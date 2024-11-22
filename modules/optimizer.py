import random
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model_selector import MODEL_MAP, PARAM_DICT_MAP

def optimize_pipeline(X, y, model_name, fs_methods, n_trials=50, cv_folds=5, scoring='accuracy'):
    """
    Optimize feature selection and model hyperparameters using Optuna.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    - model_name (str): Model to evaluate (e.g., ["KNN"]).
    - fs_methods (list): List of feature selection methods to evaluate (e.g., ["tree", "pca"]).
    - n_trials (int): Number of optimization trials.
    - cv_folds (int): Number of cross-validation folds.
    - scoring (str): Scoring metric for cross-validation.

    Returns:
    - study.best_trial: The best trial from the optimization study.
    """
    def objective(trial):
        # Select a feature selection method
        fs_method = trial.suggest_categorical("fs_method", fs_methods)

        if fs_method == "tree":
            n_estimators = trial.suggest_int("selector__n_estimators", 10, 500, log=True)
            selector = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators))
        elif fs_method == "pca":
            n_components = trial.suggest_int("selector__n_components", 5, X.shape[1])
            selector = PCA(n_components=n_components)
        elif fs_method == "univariate":
            k = trial.suggest_int("selector__k", 5, X.shape[1])
            selector = SelectKBest(mutual_info_classif, k=k)
        elif fs_method == "l1":
            C = trial.suggest_float("selector__C", 0.01, 1.0, log=True)
            selector = SelectFromModel(LogisticRegression(solver='liblinear', penalty='l1', C=C))
        else:
            raise ValueError(f"Unknown feature selection method: {fs_method}. Please select from tree, pca, univariate or l1")
        
        if isinstance(selector, (PCA, SelectKBest)):
            # Simulate fit-transform to check the output shape
            X_transformed = selector.fit_transform(X, y)
            if X_transformed.shape[1] == 0:
                print(f"Trial {trial.number} pruned due to no features left after selection.")
                raise optuna.TrialPruned()
        
        # Get the model and its hyperparameter search space
        model, param_dict = MODEL_MAP[model_name], PARAM_DICT_MAP[model_name]

        # Sample model hyperparameters dynamically
        model_params = {}
        for param_name, param_values in param_dict.items():

            if isinstance(param_values[0], int):
                model_params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))

            elif isinstance(param_values[0], float):
                if (max(param_values) / min(param_values)) >= 10:
                    log_flag = True
                else:
                    log_flag = False

                model_params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values), log=log_flag)

            elif isinstance(param_values[0], str):
                model_params[param_name] = trial.suggest_categorical(param_name, param_values)

            else:
                model_params[param_name] = random.choice(param_values)
        
        model.set_params(**model_params)
        
        if model_name == 'SVM':
            pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("model", model)
        ])
        else:
            pipeline = Pipeline([
                ("selector", selector),
                ("model", model)
            ])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, error_score='raise')
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
