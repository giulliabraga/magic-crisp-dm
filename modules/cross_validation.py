import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
import statistics as st

def cross_validation(X, y, pipelines, n_folds=10):
    features = X.values
    target = y.values

    for model_name, pipeline in pipelines.items():

        # Lists to store the metrics
        metrics_per_split = {
            'model_name': [],
            'fold': [],
            'ACSA': [],
            'recall': [],
            'CM': [],
            'f1_score': [],
            'training_time': [],
            'inference_time': [],
            'error_rate': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'precision': [],
            
        }

        print(f'Model {model_name}')

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(features, target)):
        
            print(f'Fold: {fold}')

            # Train and test split for this particular fold
            X_train_split, y_train_split = features[train_idx], target[train_idx]
            X_test_split, y_test_split = features[test_idx], target[test_idx]

            # Fitting pipeline
            start_train = time.time()
            pipeline.fit(X_train_split, y_train_split)
            stop_train = time.time()
            training_time = stop_train - start_train

            # Train and test predictions
            y_train_pred_split = pipeline.predict(X_train_split)
            start_test = time.time()
            y_test_pred_split = pipeline.predict(X_test_split)
            stop_test = time.time()
            inference_time = stop_test - start_test

            # Calculate metrics
            train_accuracy = accuracy_score(y_train_split, y_train_pred_split)
            test_accuracy = accuracy_score(y_test_split, y_test_pred_split)
            f1 = f1_score(y_test_split, y_test_pred_split)
            prec = precision_score(y_test_split, y_test_pred_split)
            rec = recall_score(y_test_split, y_test_pred_split)
            error_rate = 1 - test_accuracy  # Error rate is 1 - accuracy
            rec = recall_score(y_test_split, y_test_pred_split)
            conf_matrix = confusion_matrix(y_test_split, y_test_pred_split)
            class_accuracies = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
            acsa = class_accuracies.mean()

            # Storing metrics
            metrics_per_split['train_accuracy'].append(train_accuracy)
            metrics_per_split['test_accuracy'].append(test_accuracy)
            metrics_per_split['f1_score'].append(f1)
            metrics_per_split['precision'].append(prec)
            metrics_per_split['error_rate'].append(error_rate)
            metrics_per_split['recall'].append(rec)
            metrics_per_split['fold'].append(f'fold_{fold}')
            metrics_per_split['model_name'].append(model_name)
            metrics_per_split['ACSA'].append(acsa)
            metrics_per_split['CM'].append(conf_matrix)
            metrics_per_split['training_time'].append(training_time)
            metrics_per_split['inference_time'].append(inference_time)
            
        metrics = pd.DataFrame(metrics_per_split)
        metrics.to_csv(f'../metrics/metrics_{model_name}_ADASYN_cv.csv', index=False)

        print(f'\n Metrics: \n{metrics}')
    
    return metrics