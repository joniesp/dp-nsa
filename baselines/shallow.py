"""Shallow baselines of the paper: Isolation Forest, OC-SVM and XGBoost.

Every wrapper returns an anomaly score where higher means more anomalous.
"""
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

SEED = 1337


class IsolationForestWrapper:
    def __init__(self, n_estimators=400, random_state=SEED):
        self.clf = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    def fit(self, X): self.clf.fit(X); return self
    def score(self, X): return -self.clf.score_samples(X)

class OCSVMWrapper:
    def __init__(self, nu=0.01, gamma="scale"):
        self.oc = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    def fit(self, X): self.oc.fit(X); return self
    def score(self, X): return -self.oc.decision_function(X).ravel()


class XGBoostBalancedWrapper:
    """Supervised XGBoost with scale_pos_weight = #negatives / #positives of the training set."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train):
        num_negatives = np.sum(y_train == 0)
        num_positives = np.sum(y_train == 1)
        scale_pos = num_negatives / num_positives if num_positives > 0 else 1.0

        self.model = XGBClassifier(
            scale_pos_weight=scale_pos,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X_test):
        """Probability of the anomaly class."""
        return self.model.predict_proba(X_test)[:, 1]

    def predict(self, X_test):
        return self.model.predict(X_test)
