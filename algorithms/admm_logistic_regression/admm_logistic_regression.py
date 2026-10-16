"""
Federated logistic regression via ADMM consensus optimization.

Adapted from mdw-nl/20kChallengeVantage6 (itself based on the "Personal
Health Train" distributed learning approach described in Deist et al.,
*Distributed learning on 20 000+ lung cancer patients*), and restructured to
match this repo's single-module algorithm layout (see logistic_regression.py
for the sibling FedAvg-based implementation).

Unlike logistic_regression.py's federated averaging of locally-trained
weights, ADMM has each node solve a penalized local optimization against a
shared consensus vector ``z`` every round, which the central node updates in
closed form (plus L1 shrinkage via ``lambda_``). This converges to the same
coefficients as a centralized/pooled fit on the combined data, which plain
weight-averaging does not guarantee.

This algorithm only does federated learning: it assumes ``feature_cols`` are
already numeric and ``target_col`` is already a binary (0/1) column, present
under the exact same names on every node's data. It does not encode
categories, derive outcomes, or otherwise clean data -- that's a
data-preparation concern for the caller (e.g. ``data/datavalgen``), not this
algorithm's. Passing columns that aren't already clean numeric/binary will
produce wrong results silently rather than a validation error.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.tools.util import error, info, warn

np.random.seed(67)


# ── Column validation and train/validation split ─────────────────────────────


def _require_columns(df: pd.DataFrame, columns: list, method_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        msg = (
            f"{method_name}: column(s) {missing} not found in dataset; "
            f"available columns: {list(df.columns)}"
        )
        error(msg)
        raise ValueError(msg)


def _split_train_val(
    df: pd.DataFrame, feature_cols: list, target_col: str, train_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_clean = df.dropna(subset=[*feature_cols, target_col])
    train_df = df_clean.sample(frac=train_ratio, random_state=seed)
    val_df = df_clean.drop(train_df.index)
    return train_df, val_df


# ── Model: sigmoid / predicted probabilities ─────────────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _predict_proba(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    return _sigmoid(X_design @ coef)


# ── ADMM objectives ───────────────────────────────────────────────────────────


def _logistic_admm_objective(
    x: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    rho: float,
    features: np.ndarray,
    outcome: np.ndarray,
    total_patients: int,
) -> tuple[float, np.ndarray]:
    """Local X-update objective: logistic deviance plus the ADMM consensus penalty."""
    n_local = features.shape[0]
    X_design = np.hstack([np.ones((n_local, 1)), features])
    logits = X_design @ x
    probs = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500))), 1e-15, 1.0 - 1e-15)

    val_logistic = (-2.0 / total_patients) * np.sum(outcome * np.log(probs) + (1.0 - outcome) * np.log(1.0 - probs))
    diff = x - z + u
    val_quad = 0.5 * rho * np.dot(diff, diff)
    value = val_logistic + val_quad

    grad_sum = np.zeros_like(x)
    for i in range(n_local):
        xi = X_design[i, :]
        yi = outcome[i]
        exp_term = np.exp(np.clip(xi @ x, -500, 500))
        grad_sum += xi * (yi + (yi - 1) * exp_term) / (1 + exp_term)

    grad_logistic = (-2.0 / total_patients) * grad_sum
    grad_quad = rho * diff
    return value, grad_logistic + grad_quad


def _z_objective(
    z: np.ndarray,
    x_hat_mean: np.ndarray,
    u_mean: np.ndarray,
    rho: float,
    lambda_: float,
    num_sites: int,
) -> tuple[float, np.ndarray]:
    """Global Z-update objective: L1 shrinkage (intercept excluded) plus the consensus penalty."""
    z_reg = z.copy()
    z_reg[0] = 0.0  # Skip intercept penalty

    l1_val = lambda_ * np.linalg.norm(z_reg, 1)
    diff = z - x_hat_mean - u_mean
    quad_val = (num_sites * rho / 2.0) * np.dot(diff, diff)
    value = l1_val + quad_val

    # abs_z is floored at eps for every component (not just the intercept):
    # any z_reg entry can land on exactly 0.0 during optimization, and an
    # unguarded 0/0 there produces NaN that poisons the gradient regardless
    # of lambda_ (0.0 * NaN is NaN, not 0, under IEEE 754).
    eps = 1e-64
    abs_z = np.where(np.abs(z_reg) < eps, eps, np.abs(z_reg))
    grad_l1 = lambda_ * (z_reg / abs_z)  # grad_l1[0] == 0.0 already, since z_reg[0] == 0.0.
    grad_quad = num_sites * rho * diff
    return value, grad_l1 + grad_quad


def _check_convergence(
    x_mat: np.ndarray,
    z_vec: np.ndarray,
    z_old_vec: np.ndarray,
    u_mat: np.ndarray,
    rho: float,
    abs_tol: float,
    rel_tol: float,
) -> tuple[float, float, float, float]:
    """Primal/dual residuals and tolerances (Boyd et al. ADMM stopping criterion)."""
    num_features, num_sites = x_mat.shape
    z_mat = np.tile(z_vec.reshape(-1, 1), (1, num_sites))
    z_old_mat = np.tile(z_old_vec.reshape(-1, 1), (1, num_sites))

    r_norm = np.linalg.norm(x_mat - z_mat, ord="fro")
    s_norm = np.linalg.norm(rho * (z_mat - z_old_mat), ord="fro")

    eps_pri = np.sqrt(num_sites * num_features) * abs_tol + rel_tol * max(
        np.linalg.norm(x_mat, "fro"), np.linalg.norm(z_mat, "fro")
    )
    eps_dual = np.sqrt(num_sites * num_features) * abs_tol + rel_tol * np.linalg.norm(rho * u_mat, "fro")

    info(f"Convergence info: r_norm - eps_pri = {r_norm - eps_pri:.4e}, s_norm - eps_dual = {s_norm - eps_dual:.4e}")
    return r_norm, s_norm, eps_pri, eps_dual


def _record_training_metrics(
    history: dict[str, Any],
    site_obj: list[float],
    reg_obj: float,
    site_sse: list[float],
    z: np.ndarray,
    rho: float,
    patient_counts: list[int],
) -> None:
    total_patients = int(sum(patient_counts))
    obj_loss = float(np.sum(site_obj))
    history.setdefault("obj_loss_log", []).append(obj_loss)
    history.setdefault("obj_reg_log", []).append(reg_obj)
    history.setdefault("obj_log", []).append(obj_loss + reg_obj)

    sum_square_error = float(np.sum(site_sse))
    history.setdefault("sum_square_error_log", []).append(sum_square_error)
    history.setdefault("root_mean_square_error_log", []).append(sum_square_error / total_patients)
    history.setdefault("z_log", []).append(z.tolist())
    history.setdefault("rho_log", []).append(float(rho))


def _compute_roc_and_calibration(y_all: np.ndarray, probs_all: np.ndarray) -> dict[str, Any]:
    """Global ROC/AUC and a logistic recalibration (logit(p) = a + b * LP), analogous to
    the standalone evaluate_final_model() in the original single-file simulation."""
    eps = 1e-15
    probs_clipped = np.clip(probs_all, eps, 1.0 - eps)

    fpr_g, tpr_g, _ = roc_curve(y_all, probs_clipped)
    auc_g = roc_auc_score(y_all, probs_clipped)

    lp = np.log(probs_clipped / (1.0 - probs_clipped))
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
    lr.fit(lp.reshape(-1, 1), y_all)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])

    n_quantiles = 10
    quantile_edges = np.quantile(probs_clipped, np.linspace(0, 1, n_quantiles + 1))
    mean_pred: list[float] = []
    mean_obs: list[float] = []
    bin_centers: list[float] = []
    for q in range(n_quantiles):
        lo, hi = quantile_edges[q], quantile_edges[q + 1]
        mask = (
            (probs_clipped >= lo) & (probs_clipped <= hi)
            if q == n_quantiles - 1
            else (probs_clipped >= lo) & (probs_clipped < hi)
        )
        if not np.any(mask):
            continue
        mean_pred.append(float(np.mean(probs_clipped[mask])))
        mean_obs.append(float(np.mean(y_all[mask])))
        bin_centers.append(float(np.mean([lo, hi])))

    return {
        "roc_global": {"fpr": fpr_g.tolist(), "tpr": tpr_g.tolist(), "auc": float(auc_g)},
        "calibration": {
            "intercept": intercept,
            "slope": slope,
            "bin_centers": bin_centers,
            "mean_pred": mean_pred,
            "mean_obs": mean_obs,
        },
    }


# ── Central orchestration ────────────────────────────────────────────────────


@algorithm_client
def central(
    client: AlgorithmClient,
    feature_cols: list,
    target_col: str,
    num_rounds: int = 100,
    rho: float = 3,
    alpha: float = 1.4,
    lambda_: float = 0.0,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Orchestrate ADMM logistic regression across all nodes: compute a federated
    mean/std to normalize feature_cols, then alternate local X-updates per
    site with a closed-form global Z/U consensus update, checking primal/dual
    convergence each round, then report the final coefficients plus
    convergence history and validation metrics.
    """
    organizations = client.organization.list()
    org_ids = [organization.get("id") for organization in organizations]
    num_sites = len(org_ids)

    if num_sites == 0:
        warn("No organizations found in collaboration - nothing to do.")
        return {"status": "no_organizations"}

    num_features = len(feature_cols)
    total_features = num_features + 1  # +1 for intercept

    info("=" * 60)
    info("FEDERATED ADMM LOGISTIC REGRESSION")
    info("=" * 60)
    info(f"Organizations : {num_sites}")
    info(f"Features      : {feature_cols}")
    info(f"Target        : {target_col}")
    info(f"Rounds        : {num_rounds}  |  rho={rho}  alpha={alpha}  lambda_={lambda_}")

    # ── Phase 1: federated normalization statistics ──────────────────────────
    stats_task = client.task.create(
        input_={
            "method": "compute_stats",
            "kwargs": {
                "feature_cols": feature_cols,
                "target_col": target_col,
                "train_ratio": train_ratio,
                "seed": seed,
            },
        },
        organizations=org_ids,
        name="admm_logreg_stats",
        description="Compute local feature sums and training-row counts for global normalization",
    )
    stats_results = client.wait_for_results(task_id=stats_task.get("id"))

    patient_counts = [int(r["n"]) for r in stats_results]
    total_patients = int(sum(patient_counts))
    if total_patients == 0:
        return {"error": "No usable data across any organization"}

    global_sum = np.array([r["sum"] for r in stats_results]).sum(axis=0)
    global_sum_sq = np.array([r["sum_sq"] for r in stats_results]).sum(axis=0)
    global_mean = global_sum / total_patients
    global_var = global_sum_sq / total_patients - global_mean**2
    global_std = np.sqrt(np.maximum(global_var, 1e-8))
    info(f"total_features={total_features}, patient_counts={patient_counts}, total_patients={total_patients}")
    for feat, mu, sigma in zip(feature_cols, global_mean, global_std):
        info(f"  {feat}: mean={mu:.4f}, std={sigma:.4f}")

    # ── Phase 2: ADMM main loop ───────────────────────────────────────────────
    z = np.ones(total_features)
    u_mat = np.zeros((total_features, num_sites))
    x_mat = np.zeros((total_features, num_sites))

    history: dict[str, list[Any]] = {
        "round": [], "r_norm": [], "s_norm": [], "eps_pri": [], "eps_dual": [],
        "val_acc_mean": [], "val_acc_per_node": [], "val_rmse": [],
    }

    for r in range(1, num_rounds + 1):
        info(f"--- ADMM round {r} ---")

        tasks = [
            client.task.create(
                input_={
                    "method": "admm_x_update_partial",
                    "kwargs": {
                        "feature_cols": feature_cols,
                        "target_col": target_col,
                        "z": z.tolist(),
                        "u": u_mat[:, site_idx].tolist(),
                        "x_prev": x_mat[:, site_idx].tolist(),
                        "rho": float(rho),
                        "total_patients": total_patients,
                        "global_mean": global_mean.tolist(),
                        "global_std": global_std.tolist(),
                        "train_ratio": train_ratio,
                        "seed": seed,
                    },
                },
                organizations=[org_id],
                name=f"ADMM X-update round {r}",
                description="Local logistic regression X-update",
            )
            for site_idx, org_id in enumerate(org_ids)
        ]
        local_results = [client.wait_for_results(task_id=task.get("id"))[0] for task in tasks]

        site_sse: list[float] = []
        site_obj: list[float] = []
        for site_idx, res in enumerate(local_results):
            x_site = np.asarray(res["x"], dtype=float).reshape(-1)
            if x_site.size != total_features:
                warn(
                    f"Site {site_idx} returned {x_site.size} coefficients, "
                    f"expected {total_features}; truncating/padding as needed."
                )
                if x_site.size > total_features:
                    x_site = x_site[:total_features]
                else:
                    tmp = np.zeros(total_features)
                    tmp[: x_site.size] = x_site
                    x_site = tmp
            x_mat[:, site_idx] = x_site
            site_sse.append(float(res.get("sum_square_error", 0.0)))
            site_obj.append(float(res.get("obj", 0.0)))

        z_old = z.copy()
        z_mat = np.tile(z.reshape(-1, 1), (1, num_sites))
        x_hat = alpha * x_mat + (1.0 - alpha) * z_mat
        x_hat_mean = np.mean(x_hat, axis=1)
        u_mean = np.mean(u_mat, axis=1)

        res_z = minimize(
            fun=_z_objective,
            x0=z,
            args=(x_hat_mean, u_mean, float(rho), float(lambda_), num_sites),
            jac=True,
            method="BFGS",
            options={"disp": False},
        )
        z = res_z.x

        z_reg = z.copy()
        z_reg[0] = 0.0
        reg_obj = float(lambda_ * np.linalg.norm(z_reg, 1))

        for site_idx in range(num_sites):
            u_mat[:, site_idx] += x_hat[:, site_idx] - z

        r_norm, s_norm, eps_pri, eps_dual = _check_convergence(
            x_mat=x_mat, z_vec=z, z_old_vec=z_old, u_mat=u_mat,
            rho=float(rho), abs_tol=float(abs_tol), rel_tol=float(rel_tol),
        )

        eval_tasks = [
            client.task.create(
                input_={
                    "method": "evaluate_global_model",
                    "kwargs": {
                        "feature_cols": feature_cols,
                        "target_col": target_col,
                        "z": z.tolist(),
                        "global_mean": global_mean.tolist(),
                        "global_std": global_std.tolist(),
                        "train_ratio": train_ratio,
                        "seed": seed,
                    },
                },
                organizations=[org_id],
                name=f"ADMM evaluation round {r}",
                description="Evaluate global model on local validation data",
            )
            for org_id in org_ids
        ]
        val_acc_per_node: list[float] = []
        val_sse_per_node: list[float] = []
        val_patients_per_node: list[int] = []
        for task in eval_tasks:
            res_eval = client.wait_for_results(task_id=task.get("id"))[0]
            val_acc_per_node.append(float(res_eval["val_acc"]))
            val_sse_per_node.append(float(res_eval["val_sum_square_error"]))
            val_patients_per_node.append(int(res_eval["val_patient_count"]))

        mean_val_acc = float(np.mean(val_acc_per_node))
        total_val_sse = float(np.sum(val_sse_per_node))
        total_val_patients = int(np.sum(val_patients_per_node))
        val_rmse_global = (
            float(np.sqrt(total_val_sse / total_val_patients)) if total_val_patients > 0 else float("nan")
        )

        history["round"].append(r)
        history["r_norm"].append(float(r_norm))
        history["s_norm"].append(float(s_norm))
        history["eps_pri"].append(float(eps_pri))
        history["eps_dual"].append(float(eps_dual))
        history["val_acc_per_node"].append(val_acc_per_node)
        history["val_acc_mean"].append(mean_val_acc)
        history["val_rmse"].append(val_rmse_global)
        _record_training_metrics(
            history=history, site_obj=site_obj, reg_obj=reg_obj, site_sse=site_sse,
            z=z, rho=rho, patient_counts=patient_counts,
        )

        info(
            f"Round {r:03d}: mean val acc={mean_val_acc:.4f}, val RMSE={val_rmse_global:.4f} | "
            f"r_norm={r_norm:.4e} (eps={eps_pri:.4e}) | s_norm={s_norm:.4e} (eps={eps_dual:.4e})"
        )
        if r_norm < eps_pri and s_norm < eps_dual:
            info(f"*** ADMM converged at round {r} ***")
            break

    # ── Phase 3: final ROC/AUC and calibration (validation set only) ─────────
    info("Collecting final predictions for ROC/AUC and calibration")
    pred_tasks = [
        client.task.create(
            input_={
                "method": "collect_predictions",
                "kwargs": {
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                    "z": z.tolist(),
                    "global_mean": global_mean.tolist(),
                    "global_std": global_std.tolist(),
                    "train_ratio": train_ratio,
                    "seed": seed,
                },
            },
            organizations=[org_id],
            name="Collect predictions for ROC/calibration",
            description="Return y and predicted probabilities (validation set only)",
        )
        for org_id in org_ids
    ]

    y_all_list: list[np.ndarray] = []
    probs_all_list: list[np.ndarray] = []
    roc_site: list[dict[str, Any]] = []
    for site_idx, task in enumerate(pred_tasks):
        res_pred = client.wait_for_results(task_id=task.get("id"))[0]
        y_site = np.asarray(res_pred["y_all"], dtype=float)
        probs_site = np.asarray(res_pred["probs_all"], dtype=float)
        y_all_list.append(y_site)
        probs_all_list.append(probs_site)

        # Skipped for sites with no validation patients (e.g. train_ratio=1.0
        # or an unlucky random split on a small node).
        if y_site.size == 0:
            continue
        probs_clipped = np.clip(probs_site, 1e-15, 1.0 - 1e-15)
        fpr_s, tpr_s, _ = roc_curve(y_site, probs_clipped)
        roc_site.append({
            "site": site_idx,
            "fpr": fpr_s.tolist(),
            "tpr": tpr_s.tolist(),
            "auc": float(roc_auc_score(y_site, probs_clipped)),
        })

    y_all = np.concatenate(y_all_list) if y_all_list else np.array([])
    probs_all = np.concatenate(probs_all_list) if probs_all_list else np.array([])

    roc_global: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    if y_all.size > 0:
        metrics = _compute_roc_and_calibration(y_all=y_all, probs_all=probs_all)
        roc_global = metrics["roc_global"]
        calibration = metrics["calibration"]

    info("ADMM finished; returning final global coefficients, history and metrics")
    return {
        "coefficients": z.tolist(),
        "feature_cols": feature_cols,
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
        "history": history,
        "rho": float(rho),
        "lambda_": float(lambda_),
        "alpha": float(alpha),
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
        "patient_counts": patient_counts,
        "roc_site": roc_site,
        "roc_global": roc_global,
        "calibration": calibration,
    }


# ── Node-side (partial) functions ────────────────────────────────────────────


@data(1)
def compute_stats(
    df1: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Local training-row count plus feature sums/sum-of-squares, used by
    central() to compute the global mean/std for normalization."""
    _require_columns(df1, [*feature_cols, target_col], "compute_stats")
    train_df, _ = _split_train_val(df1, feature_cols, target_col, train_ratio, seed)
    n = len(train_df)
    info(f"Dataset: {len(df1)} total rows, {n} usable training rows after cleaning")
    if n == 0:
        return {"n": 0, "sum": [0.0] * len(feature_cols), "sum_sq": [0.0] * len(feature_cols)}

    feat_sum = train_df[feature_cols].sum().tolist()
    feat_sum_sq = (train_df[feature_cols] ** 2).sum().tolist()
    return {"n": n, "sum": feat_sum, "sum_sq": feat_sum_sq}


@data(1)
def admm_x_update_partial(
    df1: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    z: Any,
    u: Any,
    rho: float,
    total_patients: int,
    global_mean: list,
    global_std: list,
    train_ratio: float = 0.8,
    seed: int = 42,
    x_prev: Any | None = None,
) -> dict[str, Any]:
    """Local X-update for one ADMM round: minimize local logistic loss plus the
    consensus penalty tying this site's (normalized-feature) coefficients to
    the global z."""
    _require_columns(df1, [*feature_cols, target_col], "admm_x_update_partial")
    train_df, _ = _split_train_val(df1, feature_cols, target_col, train_ratio, seed)

    mean = np.asarray(global_mean, dtype=float)
    std = np.asarray(global_std, dtype=float)
    X_train = (train_df[feature_cols].to_numpy(dtype=float) - mean) / std
    y_train = train_df[target_col].to_numpy(dtype=float)

    z_arr = np.asarray(z, dtype=float).reshape(-1)
    u_arr = np.asarray(u, dtype=float).reshape(-1)

    res = minimize(
        fun=_logistic_admm_objective,
        x0=x_prev,
        args=(z_arr, u_arr, float(rho), X_train, y_train, int(total_patients)),
        jac=True,
        method="BFGS",
        options={"disp": False},
    )
    x_new = res.x

    probs = _predict_proba(X_train, x_new)
    preds = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == y_train)) if len(y_train) > 0 else 0.0
    info(f"Local training accuracy: {acc:.4f}")

    residuals = y_train - probs
    return {
        "x": x_new.tolist(),
        "patient_count": int(X_train.shape[0]),
        "train_acc": acc,
        "sum_square_error": float(np.sum(residuals**2)),
        "obj": float(res.fun),
    }


@data(1)
def evaluate_global_model(
    df1: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    z: Any,
    global_mean: list,
    global_std: list,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the current global consensus vector z on this node's validation data."""
    _require_columns(df1, [*feature_cols, target_col], "evaluate_global_model")
    _, val_df = _split_train_val(df1, feature_cols, target_col, train_ratio, seed)

    mean = np.asarray(global_mean, dtype=float)
    std = np.asarray(global_std, dtype=float)
    X_val = (val_df[feature_cols].to_numpy(dtype=float) - mean) / std
    y_val = val_df[target_col].to_numpy(dtype=float)

    probs = _predict_proba(X_val, np.asarray(z, dtype=float).reshape(-1))
    preds = (probs >= 0.5).astype(int)
    val_acc = float(np.mean(preds == y_val)) if len(y_val) > 0 else 0.0
    info(f"Validation accuracy for global model on this node: {val_acc:.4f}")

    residuals_val = y_val - probs
    return {
        "val_acc": val_acc,
        "val_sum_square_error": float(np.sum(residuals_val**2)),
        "val_patient_count": int(y_val.shape[0]),
    }


@data(1)
def collect_predictions(
    df1: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    z: Any,
    global_mean: list,
    global_std: list,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Any]:
    """Return outcomes and predicted probabilities on the validation set only,
    used by central() after convergence to compute ROC/AUC and calibration."""
    _require_columns(df1, [*feature_cols, target_col], "collect_predictions")
    _, val_df = _split_train_val(df1, feature_cols, target_col, train_ratio, seed)

    mean = np.asarray(global_mean, dtype=float)
    std = np.asarray(global_std, dtype=float)
    X_val = (val_df[feature_cols].to_numpy(dtype=float) - mean) / std
    y_val = val_df[target_col].to_numpy(dtype=float)

    probs_val = _predict_proba(X_val, np.asarray(z, dtype=float).reshape(-1))
    return {"y_all": y_val.tolist(), "probs_all": probs_val.tolist()}
