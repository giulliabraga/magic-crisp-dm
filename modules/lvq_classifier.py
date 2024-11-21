import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LVQ(BaseEstimator, ClassifierMixin):
    def __init__(self, n_codebooks=10, lrate=0.3, epochs=50):
        self.n_codebooks = n_codebooks
        self.lrate = lrate
        self.epochs = epochs

    def euclidean_distance(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2)**2))

    def get_best_matching_unit(self, codebooks, test_row):
        distances = [self.euclidean_distance(codebook[:-1], test_row) for codebook in codebooks]
        return codebooks[np.argmin(distances)]

    def train_codebooks(self, X, y):
        n_classes = len(np.unique(y))
        codebooks = np.array([np.append(X[np.random.choice(len(X))], y[np.random.choice(len(y))]) for _ in range(self.n_codebooks)])
        for epoch in range(self.epochs):
            rate = self.lrate * (1.0 - (epoch / float(self.epochs)))
            for row, label in zip(X, y):
                bmu = self.get_best_matching_unit(codebooks, row)
                for i in range(len(row)):
                    error = row[i] - bmu[i]
                    if bmu[-1] == label:
                        bmu[i] += rate * error
                    else:
                        bmu[i] -= rate * error
        return codebooks

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.codebooks_ = self.train_codebooks(X, y)
        return self

    def predict(self, X):
        predictions = [self.get_best_matching_unit(self.codebooks_, row)[-1] for row in X]
        return np.array(predictions)

    def predict_proba(self, X):
        proba = []
        for x in X:
            distances = []
            classes = []
            # Calcula a distância para cada codebook
            for codebook in self.codebooks_:
                dist = self.euclidean_distance(codebook[:-1], x)
                distances.append(dist)
                classes.append(codebook[-1])
            distances = np.array(distances)
            classes = np.array(classes)
            unique_classes = self.classes_
            # Para cada classe, encontra a menor distância
            class_scores = []
            for cls in unique_classes:
                cls_distances = distances[classes == cls]
                if len(cls_distances) == 0:
                    min_dist = np.inf  # Se não houver codebooks dessa classe
                else:
                    min_dist = np.min(cls_distances)
                class_scores.append((cls, min_dist))
            # Converte distâncias em probabilidades inversas
            total = sum(1.0 / (score[1] + 1e-10) for score in class_scores)  # Evita divisão por zero
            probs = []
            for cls, dist in class_scores:
                prob = (1.0 / (dist + 1e-10)) / total
                probs.append(prob)
            proba.append(probs)
        return np.array(proba)