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
        best_pipelines = {
        'KNN': Pipeline([
                ("selector", SelectFromModel(LogisticRegression(solver='liblinear', 
                                                                penalty='l1', 
                                                                C=0.335))),
                ("model", KNeighborsClassifier(n_neighbors=5,
                                                weights='distance',
                                                algorithm='auto',
                                                metric='manhattan'))
                ]),

        'LVQ': Pipeline([
                ("selector", PCA(n_components=25)),
                ("model", LVQ(n_codebooks=15,lrate=0.24,epochs=20))
                ]),

        'DTR': Pipeline([
                ("selector", SelectFromModel(RandomForestClassifier(n_estimators=112))),
                ("model", DecisionTreeClassifier(criterion='entropy',
                                                splitter='random',
                                                max_depth=11, 
                                                min_samples_split=4,
                                                min_samples_leaf=2,
                                                max_features='log2'))
                ]),

        'SVM': Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(mutual_info_classif, k=24)),
                ("model", SVC(probability=True, 
                                max_iter=1000, 
                                C=1.562,
                                kernel='rbf',
                                gamma='scale'))
                ]),

        'RF': Pipeline([
                ("selector", SelectKBest(mutual_info_classif, k=30)),
                ("model", RandomForestClassifier(n_estimators=366,
                                                criterion='entropy',
                                                max_depth=20, 
                                                min_samples_split=5,
                                                min_samples_leaf=2, 
                                                max_features='sqrt'))
                ]),

        'XGB': Pipeline([
                ("selector", SelectFromModel(LogisticRegression(solver='liblinear', 
                                                                penalty='l1', 
                                                                C=0.99))),
                ("model", GradientBoostingClassifier(learning_rate=0.09,
                                                        loss='log_loss',
                                                        n_estimators=202,
                                                        max_depth=6, 
                                                        subsample=0.72, 
                                                        criterion='squared_error'))
                ]),

        'LGBM': Pipeline([
                ("selector", SelectKBest(mutual_info_classif, k=30)),
                ("model", LGBMClassifier(n_estimators=193,
                                        learning_rate=0.098,
                                        num_leaves=58,
                                        max_depth=8,
                                        subsample=0.87))
                ]),

        'MLP': Pipeline([
                ("scaler", StandardScaler()),
                ("selector", PCA(n_components=28) ),
                ("model", MLPClassifier(hidden_layer_sizes=(100,50),
                                        activation='relu',
                                        solver='adam',
                                        alpha=0.002935,
                                        max_iter=200))
                ])
        } 

        selected_models = ['KNN', 'RF', 'XGB']

        estimators_het = [
            (f'pipeline_{i+1}', best_pipelines[model])
            for i, model in enumerate(selected_models)
        ]

        ann_models = [best_pipelines['MLP'] for _ in range(3)]

        estimators_anns = [(f'ann_{i+1}', model) for i, model in enumerate(ann_models)]

        ensembles = {
        'Heterogêneo': VotingClassifier(estimators=estimators_het, voting='soft'),
        'ANNs': VotingClassifier(estimators=estimators_anns, voting='soft')
        }

        return best_pipelines, ensembles