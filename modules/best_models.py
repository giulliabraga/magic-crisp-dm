import random
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lvq_classifier import LVQ

def models_to_cv():
        best_models = {
                'KNN': KNeighborsClassifier(n_neighbors=10,
                                                        weights='uniform',
                                                        algorithm='brute',
                                                        metric='manhattan'),

                'LVQ': LVQ(n_codebooks=15,lrate=0.2756588,epochs=32),
                'DTR': DecisionTreeClassifier(criterion='gini',
                                                        splitter='best',
                                                        max_depth=10, 
                                                        min_samples_split=2,
                                                        min_samples_leaf=3,
                                                        max_features='sqrt'),

                'SVM': SVC(probability=True, 
                                        max_iter=1000, 
                                        C=0.25868,
                                        kernel='rbf',
                                        gamma='scale'),

                'RF': RandomForestClassifier(n_estimators=366,
                                                        criterion='entropy',
                                                        max_depth=20, 
                                                        min_samples_split=2,
                                                        min_samples_leaf=3, 
                                                        max_features='sqrt'),

                'XGB': GradientBoostingClassifier(learning_rate=0.05147,
                                                                loss='log_loss',
                                                                n_estimators=374,
                                                                max_depth=6, 
                                                                subsample=0.571017, 
                                                                criterion='friedman_mse'),

                'LGBM': LGBMClassifier(n_estimators=180,
                                                learning_rate=0.0758,
                                                num_leaves=52,
                                                max_depth=13,
                                                subsample=0.987176),

                'MLP': MLPClassifier(hidden_layer_sizes=(100,50),
                                                activation='relu',
                                                solver='adam',
                                                alpha=0.005,
                                                max_iter=200)
                }  

        selected_models = ['XGB', 'LGBM']

        estimators_het = [
            (f'pipeline_{i+1}', best_models[model])
            for i, model in enumerate(selected_models)
        ]

        ann_models = [best_models['MLP'] for _ in range(2)]

        estimators_anns = [(f'ann_{i+1}', model) for i, model in enumerate(ann_models)]

        ensembles = {
        'Heterogêneo': StackingClassifier(estimators=estimators_het,final_estimator=LogisticRegression()),
        'ANNs': VotingClassifier(estimators=estimators_anns, voting='soft')
        }

        return best_models, ensembles