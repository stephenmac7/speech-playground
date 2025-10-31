import torch
import numpy as np

class KMeansTokenizer:
    def __init__(self, *, kmeans = None):
        if kmeans is None:
            self.kmeans, _ = torch.hub.load("bshall/dusted:main", "kmeans", language="english", trust_repo=True, verbose=False)
        else:
            self.kmeans = kmeans

    def tokenize_one(self, features : np.ndarray) -> np.ndarray:
        # features: (T, D)
        return self.kmeans.predict(features)

    def tokenize(self, batch_features : np.ndarray) -> np.ndarray:
        # batch_features: (B, T, D)
        B, _, D = batch_features.shape
        reshaped_features = batch_features.reshape(-1, D)
        tokens = self.tokenize_one(reshaped_features)
        return tokens.reshape(B, -1)