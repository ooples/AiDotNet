"""Classical-ML twins of the AiDotNet rows in ClassicalBench.cs (#2395).

Reads the train.csv / test.csv the C# side wrote (run `dotnet run ... -- --classical DIR` first), fits the
scikit-learn, XGBoost and LightGBM counterparts with the same settings, and writes python_classical.json with the
same row shape: median and IQR of the fit and predict times over the repeats, and the held-out quality.

    python classical.py DIR [--repeats 9] [--seed 42]

XGBoost and LightGBM are optional; a row whose library is not installed is reported as skipped, not dropped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


def quantile(values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default), as ClassicalBench.Quantile."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 2:], data[:, 0].astype(np.int64), data[:, 1]


def timed_rows(name: str, repeats: int, make: Callable[[int], object], fit: Callable[[object], None],
               predict: Callable[[object], np.ndarray] | None, score: Callable[[object, np.ndarray | None], float],
               metric: str) -> dict:
    fit_times: list[float] = []
    predict_times: list[float] = []
    value = 0.0
    for repeat in range(repeats):
        model = make(repeat)
        start = time.perf_counter()
        fit(model)
        fit_times.append(time.perf_counter() - start)
        predicted = None
        if predict is not None:
            start = time.perf_counter()
            predicted = predict(model)
            predict_times.append(time.perf_counter() - start)
        value = score(model, predicted)
    row = {
        "model": name,
        "fit_seconds_median": quantile(fit_times, 0.5),
        "fit_seconds_iqr": quantile(fit_times, 0.75) - quantile(fit_times, 0.25),
        "metric": metric,
        "value": float(value),
    }
    if predict_times:
        row["predict_seconds_median"] = quantile(predict_times, 0.5)
        row["predict_seconds_iqr"] = quantile(predict_times, 0.75) - quantile(predict_times, 0.25)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from threadpoolctl import threadpool_limits

    x_train, class_train, target_train = load(args.directory / "train.csv")
    x_test, class_test, target_test = load(args.directory / "test.csv")

    def accuracy(_: object, predicted: np.ndarray | None) -> float:
        return float(np.mean(predicted == class_test))

    def r2(_: object, predicted: np.ndarray | None) -> float:
        residual = np.sum((target_test - predicted) ** 2)
        return float(1.0 - residual / np.sum((target_test - target_test.mean()) ** 2))

    classify = (lambda m: m.fit(x_train, class_train)), (lambda m: m.predict(x_test))

    def one_thread(step: Callable[[object], object]) -> Callable[[object], object]:
        """Runs a step with OpenMP / BLAS pools held to one thread (scikit-learn's HGB has no n_jobs)."""
        def run(model: object) -> object:
            with threadpool_limits(limits=1):
                return step(model)
        return run

    def hgb(r: int) -> object:
        # HistGradientBoostingClassifier(maxBins 256, maxDepth 6, 100 rounds, lr 0.1, minSamplesLeaf 20), depth-wise:
        # sklearn's histogram GBM with the leaf limit lifted so depth binds.
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=6, max_leaf_nodes=None,
                                              min_samples_leaf=20, max_bins=255, early_stopping=False,
                                              random_state=args.seed)
    rows = [
        # AiDotNet RandomForestClassifierOptions: 100 trees, unlimited depth, bootstrap, one job.
        timed_rows("random_forest", args.repeats,
                   lambda r: RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=1),
                   *classify, accuracy, "accuracy"),
        timed_rows("random_forest_all_cores", args.repeats,
                   lambda r: RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1),
                   *classify, accuracy, "accuracy"),
        # Default threading on both sides (every processor), then one thread on both sides.
        timed_rows("hist_gradient_boosting", args.repeats, hgb, *classify, accuracy, "accuracy"),
        timed_rows("hist_gradient_boosting_1job", args.repeats, hgb,
                   one_thread(classify[0]), one_thread(classify[1]), accuracy, "accuracy"),
        # AiDotNet LogisticRegressionOptions: L-BFGS by default, as sklearn's lbfgs.
        timed_rows("logistic_regression", args.repeats,
                   lambda r: LogisticRegression(max_iter=1000),
                   *classify, accuracy, "accuracy"),
        timed_rows("ridge", args.repeats, lambda r: Ridge(alpha=1.0),
                   lambda m: m.fit(x_train, target_train), lambda m: m.predict(x_test), r2, "r2"),
        timed_rows("kmeans", args.repeats,
                   lambda r: KMeans(n_clusters=8, n_init=1, max_iter=300, tol=1e-4, random_state=args.seed + r),
                   lambda m: m.fit(x_train), None, lambda m, _: m.inertia_, "inertia"),
    ]

    # XGBoost and LightGBM, depth-wise like the rows above: one thread, then every processor.
    for suffix, jobs in (("_1job", 1), ("", -1)):
        try:
            import xgboost
            rows.append(timed_rows(f"hist_gradient_boosting{suffix}:xgboost", args.repeats,
                                   lambda r, jobs=jobs: xgboost.XGBClassifier(
                                       n_estimators=100, learning_rate=0.1, max_depth=6, tree_method="hist",
                                       max_bin=256, n_jobs=jobs, random_state=args.seed),
                                   *classify, accuracy, "accuracy"))
        except ImportError:
            rows.append({"model": f"hist_gradient_boosting{suffix}:xgboost", "skipped": "xgboost is not installed"})

        try:
            import lightgbm
            rows.append(timed_rows(f"hist_gradient_boosting{suffix}:lightgbm", args.repeats,
                                   lambda r, jobs=jobs: lightgbm.LGBMClassifier(
                                       n_estimators=100, learning_rate=0.1, max_depth=6, num_leaves=63,
                                       min_child_samples=20, max_bin=255, n_jobs=jobs, random_state=args.seed,
                                       verbose=-1),
                                   *classify, accuracy, "accuracy"))
        except ImportError:
            rows.append({"model": f"hist_gradient_boosting{suffix}:lightgbm", "skipped": "lightgbm is not installed"})

    import sklearn
    report = {
        "framework": f"scikit-learn {sklearn.__version__}",
        "repeats": args.repeats,
        "train_rows": len(class_train),
        "test_rows": len(class_test),
        "features": x_train.shape[1],
        "rows": rows,
    }
    path = args.directory / "python_classical.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[classical] report written to {path}")


if __name__ == "__main__":
    main()
