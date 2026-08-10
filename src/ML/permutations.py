import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


def permutation_feature_importance_cv(
    model, X, y, metric=roc_auc_score, 
    cv_splits=5, n_repeats=5, n_jobs=-1, random_state=None, verbose=True
):
    """
    Fast approximation of permutation feature importance using CV folds.

    Parameters
    ----------
    model : sklearn-like or xgboost model
        Must implement fit() and predict_proba() or predict().
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Labels.
    metric : callable
        Evaluation metric, default: roc_auc_score.
    cv_splits : int
        Number of CV folds.
    n_repeats : int
        Number of permutations per feature (lower = faster).
    n_jobs : int
        Parallel jobs.
    random_state : int or None
        Random seed.
    verbose : bool
        Show progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: ['feature', 'importance_mean', 'importance_std'].
    """

    rng = np.random.default_rng(random_state)
    features = X.columns
    importances = {f: [] for f in features}

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(X, y))

    # -- Function to compute permutation importance for one fold --
    def importance_for_fold(train_idx, val_idx):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # clone-like fit (no deep copy to keep speed)
        mdl = model.__class__(**getattr(model, "get_params", lambda: {})())

        mdl.fit(X_train, y_train)
        y_pred = predict_generic(mdl, X_val)
        baseline = metric(y_val, y_pred)

        fold_results = {}
        for f in features:
            scores = []
            X_perm = X_val.copy()
            for _ in range(n_repeats):
                X_perm[f] = rng.permutation(X_perm[f].values)
                y_perm = predict_generic(mdl, X_perm)
                scores.append(baseline - metric(y_val, y_perm))
            fold_results[f] = np.mean(scores)
        return fold_results

    # Parallel computation across folds
    iterator = tqdm(folds, disable=not verbose, desc="Permutation importance (CV)")
    fold_importances = Parallel(n_jobs=n_jobs)(
        delayed(importance_for_fold)(train_idx, val_idx) for train_idx, val_idx in iterator
    )

    # Aggregate across folds
    for fold_res in fold_importances:
        for f, imp in fold_res.items():
            importances[f].append(imp)

    df = pd.DataFrame({
        "feature": features,
        "importance_mean": [np.mean(importances[f]) for f in features],
        "importance_std": [np.std(importances[f]) for f in features]
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return df


def predict_generic(model, X):
    """Generic prediction for sklearn/xgboost models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "predict"):
        y_pred = model.predict(X)
        # for regression-like predict output
        if y_pred.ndim == 1:
            return y_pred
        return y_pred[:, 0]
    else:
        raise ValueError("Model must implement predict() or predict_proba().")
