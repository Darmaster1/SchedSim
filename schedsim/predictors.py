"""
CPU-burst-length prediction (Section 7.1 of the paper).

Five regression models are compared as candidate burst-length predictors.
Inputs are strictly limited to the four synthetic memory attributes; true
burst length, arrival time, priority, and scheduling outcomes are never
used as predictor inputs (leakage control).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .workload import Process

MODEL_FACTORIES = {
    "linear": lambda: LinearRegression(),
    "knn": lambda: KNeighborsRegressor(n_neighbors=5),
    "tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0),
    # SVR grid capped + bounded iterations: see Section 7.1 "known limitation".
    "svr": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.5, max_iter=20000),
}


def _feature_matrix(procs: List[Process], use_endcode: bool = True) -> np.ndarray:
    X = np.array([p.features() for p in procs])
    if not use_endcode:
        X = X[:, [0, 1, 3]]  # drop endcode (column index 2)
    return X


def _target_vector(procs: List[Process]) -> np.ndarray:
    return np.array([p.true_burst for p in procs], dtype=float)


@dataclass
class FittedPredictor:
    name: str
    scaler: StandardScaler
    model: object
    use_endcode: bool

    def predict(self, procs: List[Process]) -> np.ndarray:
        X = _feature_matrix(procs, self.use_endcode)
        Xs = self.scaler.transform(X)
        pred = self.model.predict(Xs)
        return np.maximum(pred, 1.0)  # floor at 1 time unit


def fit_predictor(name: str, train_procs: List[Process], use_endcode: bool = True) -> FittedPredictor:
    X = _feature_matrix(train_procs, use_endcode)
    y = _target_vector(train_procs)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = MODEL_FACTORIES[name]()
    model.fit(Xs, y)
    return FittedPredictor(name=name, scaler=scaler, model=model, use_endcode=use_endcode)


def evaluate_predictor(fp: FittedPredictor, procs: List[Process]) -> Dict[str, float]:
    y_true = _target_vector(procs)
    y_pred = fp.predict(procs)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-9
    r2 = 1 - ss_res / ss_tot
    # Spearman rank correlation (scheduling cares about ranking, not raw error)
    rank_true = np.argsort(np.argsort(y_true))
    rank_pred = np.argsort(np.argsort(y_pred))
    if np.std(rank_true) > 0 and np.std(rank_pred) > 0:
        spearman = float(np.corrcoef(rank_true, rank_pred)[0, 1])
    else:
        spearman = 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


def select_best_predictor(train_procs: List[Process], val_procs: List[Process]
                           ) -> Tuple[str, FittedPredictor, Dict[str, Dict[str, float]]]:
    """Fit every candidate model, score on validation RMSE, return the best."""
    all_metrics = {}
    best_name, best_fp, best_rmse = None, None, float("inf")
    for name in MODEL_FACTORIES:
        fp = fit_predictor(name, train_procs, use_endcode=True)
        m = evaluate_predictor(fp, val_procs)
        all_metrics[name] = m
        if m["rmse"] < best_rmse:
            best_rmse, best_name, best_fp = m["rmse"], name, fp
    return best_name, best_fp, all_metrics


def out_of_fold_predictions(procs: List[Process], model_name: str, n_folds: int = 5,
                             seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions for training-workload processes, used only to
    train the Q-learning agent so it never sees optimistic in-sample predictions."""
    n = len(procs)
    preds = np.zeros(n)
    kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
    idx_arr = np.arange(n)
    for train_idx, test_idx in kf.split(idx_arr):
        train_subset = [procs[i] for i in train_idx]
        test_subset = [procs[i] for i in test_idx]
        fp = fit_predictor(model_name, train_subset, use_endcode=True)
        preds[test_idx] = fp.predict(test_subset)
    return preds
