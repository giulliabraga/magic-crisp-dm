import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import lime
from lime import lime_tabular
import warnings
import shap


class CompleteXAI():
    def __init__(self, X_train, y_train, X_test, y_test, categorical_features, best_model, trained_models):
        self.trained_models = trained_models
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.categorical_features = categorical_features
        self.best_model = best_model
    
    def get_uncertainties(self):
        '''
        Calculating the epistemic and random uncertainty from a pool of models
        '''
        models_uncertainty = {
            "train": pd.DataFrame(columns=list(self.trained_models.keys())),
            "test": pd.DataFrame(columns=list(self.trained_models.keys()))
        }

        # Calculating the highest prediction probability for each model (uncertainty) on the training and test sets separately
        for name, model in self.trained_models.items():
            models_uncertainty["train"][name] = 1 - model.predict_proba(self.X_train).max(axis=1)
            models_uncertainty["test"][name] = 1 - model.predict_proba(self.X_test).max(axis=1)

        # The random uncertainty is the mean of the all instances' uncertainties and the epistemic uncertainty is the variance
        uncertainty_all = {
            "train": {
                "random": models_uncertainty["train"].mean(axis=1),
                "epistemic": models_uncertainty["train"].var(axis=1)
            },
            "test": {
                "random": models_uncertainty["test"].mean(axis=1),
                "epistemic": models_uncertainty["test"].var(axis=1)
            }
        }

        return uncertainty_all

    def pca_uncertainty_plots(self):
        '''
        Creates a visualization of the random and epistemic uncertainties for a given dataset
        '''
        pca = PCA(2)

        X_pca = {
            "train": pca.fit_transform(self.X_train),
            "test": pca.transform(self.X_test)
        }

        # aleatórica
        pca_fig, ax = plt.subplots(2, 3, figsize=(22, 8))

        ax[0, 0].plot(
            X_pca["train"][self.y_train.T.values[0] == 0][:, 0],
            X_pca["train"][self.y_train.T.values[0] == 0][:, 1],
            "bs",
            label="class=0"
        )
        ax[0, 0].plot(
            X_pca["train"][self.y_train.T.values[0] == 1][:, 0],
            X_pca["train"][self.y_train.T.values[0] == 1][:, 1],
            "g^",
            label="class=1"
        )

        ax[1, 0].plot(
            X_pca["test"][self.y_test.T.values[0] == 0][:, 0],
            X_pca["test"][self.y_test.T.values[0] == 0][:, 1],
            "bs",
            label="class=0"
        )
        ax[1, 0].plot(
            X_pca["test"][self.y_test.T.values[0] == 1][:, 0],
            X_pca["test"][self.y_test.T.values[0] == 1][:, 1],
            "g^",
            label="class=1"
        )

        scatter1 = ax[0, 1].scatter(
            X_pca["train"][:, 0],
            X_pca["train"][:, 1],
            c=self.uncertainty_all["train"]["random"]
        )

        pca_fig.colorbar(scatter1, ax=ax[0, 1])

        scatter2 = ax[1, 1].scatter(
            X_pca["test"][:, 0],
            X_pca["test"][:, 1],
            c=self.uncertainty_all["test"]["random"]
        )

        pca_fig.colorbar(scatter2, ax=ax[1, 1])

        scatter3 = ax[0, 2].scatter(
            X_pca["train"][:, 0],
            X_pca["train"][:, 1],
            c=self.uncertainty_all["train"]["epistemic"]
        )

        pca_fig.colorbar(scatter3, ax=ax[0, 2])

        scatter4 = ax[1, 2].scatter(
            X_pca["test"][:, 0],
            X_pca["test"][:, 1],
            c=self.uncertainty_all["test"]["epistemic"]
        )

        pca_fig.colorbar(scatter4, ax=ax[1, 2])

        ax[0, 0].set_title("Distribuição de classes em relação a PCA 1 e PCA 2 para o treino")
        ax[1, 0].set_title("Distribuição de classes em relação a PCA 1 e PCA 2 para o teste")
        ax[0, 0].legend()
        ax[1, 0].legend()

        ax[0, 1].set_title("Incerteza aleatórica para o pool de modelos em treino")
        ax[1, 1].set_title("Incerteza aleatórica para o pool de modelos em teste")

        ax[0, 2].set_title("Incerteza epistêmica para o pool de modelos em treino")
        ax[1, 2].set_title("Incerteza epistêmica para o pool de modelos em teste")

        return pca_fig
    
    def get_instances(self):
        X_test, y_test, best_model = self.X_test, self.y_test, self.best_model

        instances = {}

        uncertainties = 1 - best_model.predict_proba(X_test).max(axis=1)

        predictions = best_model.predict(X_test)

        is_prediction_correct = predictions == y_test

        try:
            instances["high_uncertainty_correct_prediction"] = random.choice(
                np.where((uncertainties > 0.40) & is_prediction_correct)[0]
            )
            instances["low_uncertainty_correct_prediction"] = random.choice(
                np.where((uncertainties < 0.15) & is_prediction_correct)[0]
            )
            instances["high_uncertainty_wrong_prediction"] = random.choice(
                np.where((uncertainties > 0.40) & ~is_prediction_correct)[0]
            )
            instances["low_uncertainty_wrong_prediction"] = random.choice(
                np.where((uncertainties < 0.15) & ~is_prediction_correct)[0]
            )
        except IndexError:
            pass  # Simply ignore if any condition has no matches

        return instances, predictions

    
    def init_lime(self):
        # LIME demands you identify which variables are categorical in an array of booleans
        categorical_features_bool = [col in self.categorical_features for col in self.X_test.columns]

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_test),
            feature_names=list(self.X_test.columns),
            categorical_features=categorical_features_bool,
            class_names=["0", "1"],
            random_state=42,
        )

        return lime_explainer
    
    def init_shap(self):
        shap_explainer = shap.Explainer(self.best_model)
        shap_values = shap_explainer(self.X_test)

        return shap_explainer, shap_values
    
    def run_local_explanations(self):

        lime_explainer = self.init_lime()
        shap_explainer, shap_values = self.init_shap()

        instances, predictions = self.get_instances()

        for key, instance in instances.items():
            print(f'Instância {instance}\n')
            print(f'Variáveis:\n {self.X_test.iloc[instance]}\n')
            print(f'Classe verdadeira {self.y.iloc[instance]} e classe predita {predictions[instance]}')
            print(f'Status: \n {key}')

            print('\nIniciando LIME...\n')

            warnings.filterwarnings("ignore")

            exp = lime_explainer.explain_instance(self.X_test.iloc[instance], self.best_model.predict_proba, num_features=10)
            fig_lime = exp.as_pyplot_figure()
            fig_lime.suptitle(f'Instância {instance}', fontsize=16)
            fig_lime.show()

            exp.show_in_notebook(show_table=True)

            print('Iniciando SHAP...')

            fig_waterfall_class_1 = shap.plots.waterfall(shap_values[instance,:,1], max_display=20)
            fig_waterfall_class_1.suptitle(f'Instância {instance}, waterfall para classe 1', fontsize=16)
            fig_waterfall_class_1.show()

    def run_global_explanations(self):
        shap_explainer, shap_values = self.init_shap()

        fig_barplots_class_1 = shap.plots.bar(shap_values[:,:,1].abs.mean(0))
        fig_barplots_class_1.suptitle(f'Barplot para classe 1', fontsize=16)

        fig_violin_class_1 = shap.plots.violin(shap_values[:,:,1], max_display=20,plot_type="layered_violin")
        fig_violin_class_1.suptitle(f'Violin plot para classe 1', fontsize=16)






