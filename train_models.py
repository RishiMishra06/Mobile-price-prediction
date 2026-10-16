"""Offline training for the market price model.

Run this whenever phones.csv changes:  python train_models.py
It runs GridSearchCV, evaluates with nested cross-validation (honest for a small dataset),
fits the final model, and saves market_model.joblib + model_card.json for the app.
"""
import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RepeatedKFold, cross_val_predict

import core

warnings.filterwarnings("ignore")


def main():
    cat = core.load_catalog("phones.csv")
    problems = core.validate_catalog(cat)
    if problems:
        raise SystemExit("Catalog problems: " + "; ".join(problems))
    d, X, y, _ = core._market_frame(cat)
    price = d["price_inr"].values
    print(f"{len(d)} phones")

    # 1) hyperparameter search
    gs = GridSearchCV(GradientBoostingRegressor(random_state=42), core.PARAM_GRID,
                      cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error",
                      n_jobs=-1).fit(X, y)
    best = dict(gs.best_params_, random_state=42)
    print("best params:", best)

    # 2) nested CV: outer 6-fold x 2 repeats, inner 5-fold grid search
    reps, folds = 2, 6
    oof = np.zeros((reps, len(y)))
    for i, (tr, te) in enumerate(RepeatedKFold(n_splits=folds, n_repeats=reps, random_state=7).split(X)):
        inner = GridSearchCV(GradientBoostingRegressor(random_state=42), core.PARAM_GRID,
                             cv=KFold(5, shuffle=True, random_state=1), scoring="neg_mean_absolute_error",
                             n_jobs=-1).fit(X.iloc[tr], y[tr])
        oof[i // folds, te] = inner.predict(X.iloc[te])
    per_rep = [dict(r2=r2_score(y, o), mae=mean_absolute_error(price, np.exp(o)),
                    mape=100 * np.mean(np.abs(np.exp(o) - price) / price)) for o in oof]
    print("nested CV per repeat:", per_rep)

    # 3) baseline (untuned) and quantile-regression comparison, for the model card
    default = dict(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42)
    loo_default = cross_val_predict(GradientBoostingRegressor(**default), X, y, cv=LeaveOneOut())
    qp = {k: v for k, v in best.items() if k != "subsample"}
    lo = cross_val_predict(GradientBoostingRegressor(loss="quantile", alpha=0.1, **qp), X, y, cv=LeaveOneOut())
    hi = cross_val_predict(GradientBoostingRegressor(loss="quantile", alpha=0.9, **qp), X, y, cv=LeaveOneOut())
    quantile_cov = float(np.mean((y >= lo) & (y <= hi)))

    # 4) final artifact: residuals from the nested out-of-fold predictions
    oof_mean = oof.mean(axis=0)
    method = f"nested {folds}-fold CV x{reps} with GridSearchCV"
    art = core._finalize(cat, best, oof_mean, method,
                         extra_metrics=dict(per_repeat=per_rep, grid_cv_mae_log=float(-gs.best_score_)))
    n = len(y)
    res = np.abs(y - oof_mean)
    cov = float(np.mean([abs(y[i] - oof_mean[i]) <= np.quantile(
        np.delete(res, i), min(1.0, math.ceil(n * core.COVERAGE) / (n - 1))) for i in range(n)]))
    # report the average of the per-repeat scores (averaging predictions across repeats would be optimistic)
    for k in ("r2", "mae", "mape"):
        art["metrics"][k] = float(np.mean([r[k] for r in per_rep]))
    art["metrics"]["conformal_coverage"] = cov
    art["metrics"]["quantile_coverage"] = quantile_cov
    art["metrics"]["default_loo"] = dict(
        r2=float(r2_score(y, loo_default)), mae=float(mean_absolute_error(price, np.exp(loo_default))))

    core.save_artifact(art, "market_model.joblib")
    card = dict(built_at=art["built_at"], sklearn=art["sklearn_version"], n_phones=art["n"], params=best,
                features=core.MARKET_FEATS, metrics=art["metrics"], conformal_halfwidth_factor=math.exp(art["conformal_q"]),
                catalog_hash=art["hash"])
    with open("model_card.json", "w") as f:
        json.dump(card, f, indent=2, default=float)
    print(json.dumps(card["metrics"], indent=2, default=float))


if __name__ == "__main__":
    main()
