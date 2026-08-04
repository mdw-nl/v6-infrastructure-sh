#!/usr/bin/env python3
"""Generate a synthetic BEACH dataset with a strong, validated survival signal.

Adapted from `20kChallengeVantage6/scripts/generate_signal_synthetic_dataset.py`
(mdw-nl/20kChallengeVantage6) so this repo can produce BEACH-schema node data
without depending on that repo at runtime. Uses `datavalgen` +
`datavalgen-model-beach` to create schema-valid base rows, then injects
controlled signal into outcomes and diagnosis years so a centralized logistic
model reaches a minimum target AUC.

Node names in the output split match this repo's actual node/org names
(alpha, beta, gamma, theta, ...) instead of the upstream script's generic
alpha/beta/gamma/delta/epsilon/zeta list.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURE_COLUMNS = (
    "patient_t_stage",
    "patient_n_stage",
    "patient_m_stage",
    "patient_overall_stage",
)

T_MAP = {
    "T0": 0,
    "T1": 1,
    "T1a": 1,
    "T1b": 1,
    "T1c": 1,
    "T1mi": 1,
    "Tis": 1,
    "T2": 2,
    "T2a": 2,
    "T2b": 2,
    "T3": 3,
    "T4": 4,
    "Tx": 5,
}
N_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3, "Nx": 4}
M_MAP = {"M0": 0, "M1": 1, "M1a": 1, "M1b": 1, "M1c": 1, "Mx": 2}
S_MAP = {
    "0": 0,
    "I": 1,
    "IA": 1,
    "IA1": 1,
    "IA2": 1,
    "IA3": 1,
    "IB": 1,
    "II": 2,
    "IIA": 2,
    "IIB": 2,
    "III": 3,
    "IIIA": 3,
    "IIIB": 3,
    "IIIC": 3,
    "IV": 4,
    "IVA": 4,
    "IVB": 4,
    "Occult": 5,
    "x": 5,
}


def _map_tnm(series: pd.Series, mmap: dict) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.mask(s.str.lower() == "nan", np.nan)
    return s.map(mmap)


def _map_overall(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    str_stripped = series.astype(str).str.strip()
    str_stripped = str_stripped.mask(str_stripped.str.lower() == "nan", np.nan)
    is_whole = num.notna() & (num == np.floor(num))
    keys = pd.Series(index=series.index, dtype=object)
    keys.loc[series.isna()] = np.nan
    keys.loc[is_whole & series.notna()] = num.loc[is_whole & series.notna()].astype(int).astype(str)
    keys.loc[~is_whole & series.notna()] = str_stripped.loc[~is_whole & series.notna()]
    return keys.map(S_MAP)


def _prepare_stage_columns_for_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_t_stage"] = _map_tnm(out["patient_t_stage"], T_MAP)
    out["patient_n_stage"] = _map_tnm(out["patient_n_stage"], N_MAP)
    out["patient_m_stage"] = _map_tnm(out["patient_m_stage"], M_MAP)
    out["patient_overall_stage"] = _map_overall(out["patient_overall_stage"])
    out["patient_t_stage"] = pd.Categorical(
        out["patient_t_stage"], categories=sorted(set(T_MAP.values()))
    )
    out["patient_n_stage"] = pd.Categorical(
        out["patient_n_stage"], categories=sorted(set(N_MAP.values()))
    )
    out["patient_m_stage"] = pd.Categorical(
        out["patient_m_stage"], categories=sorted(set(M_MAP.values()))
    )
    out["patient_overall_stage"] = pd.Categorical(
        out["patient_overall_stage"], categories=sorted(set(S_MAP.values()))
    )
    return out


# Risk maps are intentionally monotonic with disease severity.
T_RISK = {
    "Tx": 0.8,
    "Tis": 0.1,
    "T0": 0.2,
    "T1": 1.0,
    "T1mi": 0.9,
    "T1a": 1.0,
    "T1b": 1.3,
    "T1c": 1.6,
    "T2": 2.1,
    "T2a": 2.3,
    "T2b": 2.8,
    "T3": 3.8,
    "T4": 4.9,
}
N_RISK = {"Nx": 0.5, "N0": 0.0, "N1": 1.8, "N2": 3.2, "N3": 4.4}
M_RISK = {"Mx": 0.7, "M0": 0.0, "M1": 2.5, "M1a": 2.9, "M1b": 3.7, "M1c": 4.6}
S_RISK = {
    "0": 0.0,
    "IA1": 0.5,
    "IA2": 0.8,
    "IA3": 1.1,
    "IB": 1.4,
    "IIA": 2.0,
    "IIB": 2.4,
    "IIIA": 3.1,
    "IIIB": 3.5,
    "IIIC": 3.9,
    "IVA": 4.4,
    "IVB": 5.0,
    "x": 2.2,
}

# Matches this repo's actual node/org names (see infrastructure/nodes.env).
NODE_NAMES = ["alpha", "beta", "gamma", "theta", "epsilon", "zeta"]
CENTRES = ["isala", "radboud", "umcg", "maastricht", "erasmus", "amc"]
ADULT_AGE_MIN = 40
ADULT_AGE_MAX = 90


@dataclass
class EvalMetrics:
    train_auc: float
    val_auc: float
    train_count: int
    val_count: int
    prevalence_alive: float


@dataclass
class SignalConfig:
    signal_scale: float
    bias: float
    noise_std: float


def _balanced_centres(n: int, rng: np.random.Generator) -> np.ndarray:
    reps = int(np.ceil(n / len(CENTRES)))
    values = np.tile(np.array(CENTRES, dtype=object), reps)[:n]
    rng.shuffle(values)
    return values


def _choice(rng: np.random.Generator, values: list[str]) -> str:
    return values[int(rng.integers(0, len(values)))]


def _derive_overall_stage(t_stage: str, n_stage: str, m_stage: str, rng: np.random.Generator) -> str:
    if m_stage in {"M1", "M1a", "M1b", "M1c"}:
        if m_stage == "M1c":
            return "IVB"
        if m_stage == "M1b":
            return _choice(rng, ["IVA", "IVB"])
        return "IVA"

    if m_stage == "Mx":
        if n_stage in {"N2", "N3"} or t_stage in {"T3", "T4"}:
            return _choice(rng, ["x", "IIIB", "IIIC"])
        return _choice(rng, ["x", "IIA", "IIB", "IIIA"])

    # m_stage == M0
    if t_stage == "Tx" and n_stage == "Nx":
        return "x"
    if n_stage == "Nx":
        if t_stage in {"T3", "T4"}:
            return _choice(rng, ["IIIA", "IIIB", "x"])
        return _choice(rng, ["IIB", "IIIA", "x"])

    if n_stage == "N3":
        return _choice(rng, ["IIIB", "IIIC"])
    if n_stage == "N2":
        if t_stage in {"T3", "T4"}:
            return _choice(rng, ["IIIB", "IIIC"])
        return _choice(rng, ["IIIA", "IIIB"])
    if n_stage == "N1":
        if t_stage in {"Tis", "T0", "T1", "T1mi", "T1a", "T1b", "T1c"}:
            return _choice(rng, ["IIB", "IIIA"])
        return _choice(rng, ["IIIA", "IIIB"])

    # n_stage == N0
    if t_stage in {"Tis", "T0"}:
        return _choice(rng, ["0", "IA1"])
    if t_stage in {"T1", "T1mi", "T1a"}:
        return _choice(rng, ["IA1", "IA2"])
    if t_stage in {"T1b", "T1c"}:
        return _choice(rng, ["IA2", "IA3", "IB"])
    if t_stage in {"T2", "T2a"}:
        return _choice(rng, ["IB", "IIA", "IIB"])
    if t_stage == "T2b":
        return _choice(rng, ["IIA", "IIB", "IIIA"])
    if t_stage == "T3":
        return _choice(rng, ["IIB", "IIIA", "IIIB"])
    if t_stage == "T4":
        return _choice(rng, ["IIIA", "IIIB"])

    return "x"


def _enforce_clinical_consistency(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    derived_stage = [
        _derive_overall_stage(t, n, m, rng)
        for t, n, m in zip(
            out["patient_t_stage"].astype(str),
            out["patient_n_stage"].astype(str),
            out["patient_m_stage"].astype(str),
        )
    ]
    out["patient_overall_stage"] = derived_stage
    return out


def _assert_dataset_quality(df: pd.DataFrame) -> None:
    # age_at_diagnosis is stored as days-since-birth per the beach_lung schema
    # (datavalgen_model_beach.model: Field(ge=18*365+5, le=130*365)).
    ages_days = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
    if ages_days.isna().any():
        raise RuntimeError("age_at_diagnosis contains non-numeric values")
    min_days = ADULT_AGE_MIN * 365
    max_days = 120 * 365
    if (ages_days < min_days).any() or (ages_days > max_days).any():
        bad = int(((ages_days < min_days) | (ages_days > max_days)).sum())
        raise RuntimeError(f"age_at_diagnosis out of expected bounds in {bad} rows")

    metastatic = df["patient_m_stage"].isin(["M1", "M1a", "M1b", "M1c"])
    stage4 = df["patient_overall_stage"].isin(["IVA", "IVB"])
    bad_metastatic = int((metastatic & ~stage4).sum())
    if bad_metastatic:
        raise RuntimeError(
            f"Found {bad_metastatic} rows with metastatic disease but non-IV overall stage"
        )


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nstdout:\n"
            + proc.stdout
            + "\n\nstderr:\n"
            + proc.stderr
        )


def _datavalgen_generate(num_rows: int, output_csv: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "datavalgen",
            "generate",
            "-f",
            "beach_lung",
            "-n",
            str(num_rows),
            "-o",
            str(output_csv),
            "--force",
        ]
    )


def _datavalgen_validate(csv_path: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "datavalgen",
            "validate",
            "-m",
            "beach_lung",
            "-d",
            str(csv_path),
        ]
    )


def _inject_signal(
    df: pd.DataFrame,
    rng: np.random.Generator,
    cfg: SignalConfig,
    train_fraction: float,
    attempt: int,
) -> pd.DataFrame:
    out = df.copy()

    n = len(out)
    out["patient_identifier"] = [f"PT_{attempt:02d}_{i:06d}" for i in range(n)]
    out["centre"] = _balanced_centres(n, rng)

    # Raw datavalgen output stores age_at_diagnosis as days-since-birth; work
    # in years for realism/severity, then convert back to days for the schema
    # (datavalgen_model_beach.model requires days, ge=18*365+5, le=130*365).
    age_years = pd.to_numeric(out["age_at_diagnosis"], errors="coerce") / 365.0
    invalid_age = age_years.isna() | (age_years < ADULT_AGE_MIN) | (age_years > ADULT_AGE_MAX)
    if invalid_age.any():
        age_years.loc[invalid_age] = rng.integers(
            ADULT_AGE_MIN, ADULT_AGE_MAX + 1, size=int(invalid_age.sum())
        )
    age_years = age_years.round().astype(int)
    out["age_at_diagnosis"] = age_years * 365

    out = _enforce_clinical_consistency(out, rng)

    severity = (
        out["patient_t_stage"].map(T_RISK).astype(float) * 0.9
        + out["patient_n_stage"].map(N_RISK).astype(float) * 1.0
        + out["patient_m_stage"].map(M_RISK).astype(float) * 1.3
        + out["patient_overall_stage"].map(S_RISK).astype(float) * 1.1
    )
    severity += np.nan_to_num((age_years - 65.0) / 20.0, nan=0.0)

    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_fraction)
    years = np.empty(n, dtype=int)
    years[idx[:n_train]] = rng.integers(1994, 2012, size=n_train)
    years[idx[n_train:]] = rng.integers(2012, 2025, size=n - n_train)
    out["year_of_diagnosis"] = years

    logit = cfg.bias - cfg.signal_scale * severity + rng.normal(0.0, cfg.noise_std, size=n)
    prob_alive = 1.0 / (1.0 + np.exp(-logit))
    alive_mask = rng.random(n) < prob_alive

    # Rare safeguard against degenerate single-class samples.
    if alive_mask.sum() == 0:
        alive_mask[np.argmin(severity)] = True
    if alive_mask.sum() == n:
        alive_mask[np.argmax(severity)] = False

    out["vital_status"] = np.where(alive_mask, "alive", "dead")

    dead_days = np.clip(
        (1600.0 - 190.0 * severity + rng.normal(0.0, 300.0, size=n)).astype(int),
        20,
        3200,
    )
    alive_days = np.clip(
        (2600.0 - 80.0 * severity + rng.normal(0.0, 300.0, size=n)).astype(int),
        250,
        5000,
    )
    out["interval_diagnosis_to_last_visit_in_days"] = np.where(
        alive_mask,
        alive_days,
        dead_days,
    )

    return out


def _evaluate_centralized(df: pd.DataFrame) -> EvalMetrics | None:
    work = df.copy()

    work = _prepare_stage_columns_for_dummies(work)

    work["SurvivalStatus"] = (
        work["vital_status"].astype(str).str.lower().str.strip() == "alive"
    ).astype(int)
    work["__diag_year__"] = pd.to_numeric(work["year_of_diagnosis"], errors="coerce")

    work = work.dropna(subset=[*FEATURE_COLUMNS, "SurvivalStatus", "__diag_year__"])

    X = pd.get_dummies(work[list(FEATURE_COLUMNS)], drop_first=True).to_numpy(dtype=float)
    y = work["SurvivalStatus"].to_numpy(dtype=int)
    years = work["__diag_year__"].to_numpy(dtype=int)

    train_mask = years <= 2011
    val_mask = years >= 2012

    if train_mask.sum() < 30 or val_mask.sum() < 30:
        return None
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2:
        return None

    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X[train_mask], y[train_mask])

    train_auc = roc_auc_score(
        y[train_mask], model.predict_proba(X[train_mask])[:, 1]
    )
    val_auc = roc_auc_score(y[val_mask], model.predict_proba(X[val_mask])[:, 1])

    return EvalMetrics(
        train_auc=float(train_auc),
        val_auc=float(val_auc),
        train_count=int(train_mask.sum()),
        val_count=int(val_mask.sum()),
        prevalence_alive=float(y.mean()),
    )


def _write_node_split(df: pd.DataFrame, split_dir: Path, node_count: int, seed: int) -> dict[str, Any]:
    node_names = NODE_NAMES[:node_count]

    work = df.copy()
    centre_counts = work["centre"].astype(str).value_counts().to_dict()
    centres_sorted = sorted(centre_counts.keys(), key=lambda c: (-centre_counts[c], c))

    node_load = [0 for _ in range(node_count)]
    centre_to_node_idx: dict[str, int] = {}
    for centre in centres_sorted:
        target_node = min(range(node_count), key=lambda idx: (node_load[idx], idx))
        centre_to_node_idx[centre] = target_node
        node_load[target_node] += int(centre_counts[centre])

    work["__node__"] = work["centre"].map(centre_to_node_idx).astype(int)
    spread = work.groupby("centre")["__node__"].nunique()
    if (spread > 1).any():
        raise RuntimeError("Center-based split violated: at least one center spans multiple nodes")
    split_dir.mkdir(parents=True, exist_ok=True)

    node_to_centres: dict[str, list[str]] = {}
    node_counts: dict[str, int] = {}
    for node_idx, node_name in enumerate(node_names):
        node_df = work[work["__node__"] == node_idx].drop(columns=["__node__"])
        node_centres = sorted(node_df["centre"].astype(str).unique().tolist())
        node_to_centres[node_name] = node_centres
        node_counts[node_name] = int(len(node_df))
        node_df.to_csv(split_dir / f"{node_name}.csv", index=False)

    return {
        "centre_to_node": {centre: node_names[idx] for centre, idx in centre_to_node_idx.items()},
        "node_to_centres": node_to_centres,
        "node_row_counts": node_counts,
        "seed": seed,
    }


def _parse_node_counts(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 3 or value > 6:
            raise ValueError(f"Node counts must be between 3 and 6, got: {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("No valid node-count values were provided")
    return values


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output = repo_root / "data" / "beach"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--num-subjects", type=int, default=2000)
    parser.add_argument("--node-counts", default="4")
    parser.add_argument("--default-node-count", type=int, default=4)
    parser.add_argument("--target-train-auc", type=float, default=0.90)
    parser.add_argument("--target-val-auc", type=float, default=0.80)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260323)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_subjects < 500:
        raise SystemExit("--num-subjects must be at least 500")
    if not (0.5 < args.train_fraction < 0.95):
        raise SystemExit("--train-fraction must be between 0.5 and 0.95")

    node_counts = _parse_node_counts(args.node_counts)
    if args.default_node_count not in node_counts:
        raise SystemExit(
            f"--default-node-count ({args.default_node_count}) must be in --node-counts ({node_counts})"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_schedule = [
        SignalConfig(0.95, 6.4, 1.0),
        SignalConfig(1.00, 6.4, 1.0),
        SignalConfig(1.05, 6.6, 1.0),
        SignalConfig(1.05, 6.8, 1.0),
        SignalConfig(1.10, 6.8, 1.0),
        SignalConfig(1.10, 7.0, 0.95),
    ]

    best_df: pd.DataFrame | None = None
    best_metrics: EvalMetrics | None = None
    best_cfg: SignalConfig | None = None
    best_attempt = -1

    with tempfile.TemporaryDirectory(prefix="beach_signal_gen_") as tmp:
        tmp_dir = Path(tmp)

        for attempt in range(args.max_attempts):
            cfg = signal_schedule[min(attempt, len(signal_schedule) - 1)]
            raw_path = tmp_dir / f"raw_{attempt:02d}.csv"

            _datavalgen_generate(args.num_subjects, raw_path)
            raw_df = pd.read_csv(raw_path)

            rng = np.random.default_rng(args.seed + 17_729 * (attempt + 1))
            candidate_df = _inject_signal(
                raw_df,
                rng=rng,
                cfg=cfg,
                train_fraction=args.train_fraction,
                attempt=attempt,
            )
            metrics = _evaluate_centralized(candidate_df)
            if metrics is None:
                print(f"attempt={attempt:02d} skipped (degenerate split)")
                continue

            print(
                "attempt={attempt:02d} "
                "train_auc={train:.4f} val_auc={val:.4f} "
                "train_n={tn} val_n={vn} alive_prev={prev:.4f} "
                "cfg(scale={scale:.2f},bias={bias:.2f},noise={noise:.2f})".format(
                    attempt=attempt,
                    train=metrics.train_auc,
                    val=metrics.val_auc,
                    tn=metrics.train_count,
                    vn=metrics.val_count,
                    prev=metrics.prevalence_alive,
                    scale=cfg.signal_scale,
                    bias=cfg.bias,
                    noise=cfg.noise_std,
                )
            )

            if (
                best_metrics is None
                or (metrics.train_auc + metrics.val_auc)
                > (best_metrics.train_auc + best_metrics.val_auc)
            ):
                best_df = candidate_df
                best_metrics = metrics
                best_cfg = cfg
                best_attempt = attempt

            if (
                metrics.train_auc >= args.target_train_auc
                and metrics.val_auc >= args.target_val_auc
            ):
                best_df = candidate_df
                best_metrics = metrics
                best_cfg = cfg
                best_attempt = attempt
                break

    if best_df is None or best_metrics is None or best_cfg is None:
        raise SystemExit("Unable to generate a dataset that could be evaluated")

    if not (
        best_metrics.train_auc >= args.target_train_auc
        and best_metrics.val_auc >= args.target_val_auc
    ):
        raise SystemExit(
            "No attempt met target AUC thresholds. Best attempt={} had "
            "train_auc={:.4f}, val_auc={:.4f}."
            .format(best_attempt, best_metrics.train_auc, best_metrics.val_auc)
        )

    full_csv = output_dir / "beach_signal_full.csv"
    best_df.to_csv(full_csv, index=False)

    _datavalgen_validate(full_csv)
    _assert_dataset_quality(best_df)

    split_outputs: dict[str, Any] = {}
    for node_count in node_counts:
        split_dir = output_dir / f"splits_{node_count}nodes"
        split_meta = _write_node_split(
            best_df, split_dir, node_count=node_count, seed=args.seed + node_count
        )
        split_outputs[str(node_count)] = {
            "split_dir": str(split_dir),
            "files": [str(split_dir / f"{name}.csv") for name in NODE_NAMES[:node_count]],
            "meta": split_meta,
        }

    report = {
        "full_dataset_csv": str(full_csv),
        "best_attempt": best_attempt,
        "num_subjects": int(len(best_df)),
        "target_train_auc": args.target_train_auc,
        "target_val_auc": args.target_val_auc,
        "metrics": {
            "train_auc": best_metrics.train_auc,
            "val_auc": best_metrics.val_auc,
            "train_count": best_metrics.train_count,
            "val_count": best_metrics.val_count,
            "prevalence_alive": best_metrics.prevalence_alive,
        },
        "signal_config": {
            "signal_scale": best_cfg.signal_scale,
            "bias": best_cfg.bias,
            "noise_std": best_cfg.noise_std,
        },
        "node_splits": split_outputs,
        "default_node_count": args.default_node_count,
        "default_split_dir": str(output_dir / f"splits_{args.default_node_count}nodes"),
    }

    report_path = output_dir / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("saved_dataset=", full_csv)
    print("saved_report=", report_path)
    print(
        "final_metrics: train_auc={:.4f} val_auc={:.4f} train_n={} val_n={} alive_prev={:.4f}".format(
            best_metrics.train_auc,
            best_metrics.val_auc,
            best_metrics.train_count,
            best_metrics.val_count,
            best_metrics.prevalence_alive,
        )
    )


if __name__ == "__main__":
    main()
