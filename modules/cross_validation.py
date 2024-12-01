import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
import statistics as st
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt


def calculate_loacc(y_true, y_scores):

    from sklearn.metrics import roc_curve
    
    # Calcula a curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Valores de aceitação do fundo específicos
    thresholds = [0.01, 0.02, 0.05]
    
    # Interpolação linear para calcular TPR nos thresholds desejados
    loacc_values = []
    for t in thresholds:
        for i in range(len(fpr) - 1):
            if fpr[i] <= t < fpr[i + 1]:  # Encontrar o intervalo [FPR[i], FPR[i+1]]
                interp_value = ((tpr[i+1] - tpr[i]) / (fpr[i+1] - fpr[i])) * (t - fpr[i]) + tpr[i]
                loacc_values.append(interp_value)
                break

    # Calcula a média dos valores de TPR encontrados
    loacc = sum(loacc_values) / len(thresholds)
    return loacc

def cross_validation(X, y, models, n_folds=10, use_adasyn=False):
    features = X.values
    target = y.values

    for model_name, model in models.items():

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
            'loacc': [],
            'auc': []
        }

        print(f'Model {model_name}')

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(features, target)):
        
            print(f'Fold: {fold}')

            # Train and test split for this particular fold
            X_train_split, y_train_split = features[train_idx], target[train_idx]
            X_test_split, y_test_split = features[test_idx], target[test_idx]

            if use_adasyn == True:
                adasyn = ADASYN(sampling_strategy='auto', random_state=42)
                X_train_split, y_train_split = adasyn.fit_resample(X_train_split, y_train_split)

            # Fitting pipeline
            start_train = time.time()
            model.fit(X_train_split, y_train_split)
            stop_train = time.time()
            training_time = stop_train - start_train

            # Train and test predictions
            y_train_pred_split = model.predict(X_train_split)
            start_test = time.time()
            y_test_pred_split = model.predict(X_test_split)
            stop_test = time.time()
            inference_time = stop_test - start_test

            y_scores = model.predict_proba(X_test_split)[:, 1]

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

            loacc = calculate_loacc(y_test_split,y_scores)

            

            fpr, tpr, _ = roc_curve(y_test_split, y_scores)
            auc_value = auc(fpr, tpr)
            '''
            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"Curva ROC")
            plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
            plt.title(f"Curva ROC (AUC = {auc_value:.2f}), modelo {model_name} e fold {fold}")
            plt.xlabel("Taxa de Falso Positivo (FPR)")
            plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(f'../outputs/roc_curve_{model_name}_fold_{fold}')
            '''

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
            metrics_per_split['loacc'].append(loacc)
            metrics_per_split['auc'].append(auc_value)
            

        metrics = pd.DataFrame(metrics_per_split)

        if use_adasyn == True:
            metrics.to_csv(f'../metrics_correct/metrics_adasyn_{model_name}_cv.csv', index=False)
        else:
            metrics.to_csv(f'../metrics_correct/metrics_{model_name}_cv.csv', index=False)

        print(f'\n Metrics: \n{metrics}')
    
    return metrics