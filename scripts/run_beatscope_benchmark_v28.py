from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


MITBIH_LABELS = {
    0: "N_normal",
    1: "S_supraventricular",
    2: "V_ventricular",
    3: "F_fusion",
    4: "Q_unknown",
}

PTBDB_LABELS = {
    0: "normal",
    1: "abnormal",
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_float(x):
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def safe_to_markdown(df: pd.DataFrame, path: Path) -> None:
    try:
        path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        path.write_text(df.to_csv(index=False), encoding="utf-8")


def find_file_case_insensitive(root: Path, name: str) -> Path | None:
    exact = root / name
    if exact.exists():
        return exact

    target = name.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def extract_zip_if_needed(raw_dir: Path, zip_name: str, csv_name: str) -> Path:
    ensure_dir(raw_dir)

    csv_path = find_file_case_insensitive(raw_dir, csv_name)
    if csv_path and csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path

    zip_path = find_file_case_insensitive(raw_dir, zip_name)
    if zip_path is None:
        raise FileNotFoundError(
            f"Missing {zip_name}. Put it under {raw_dir}. "
            f"Required files: ptbdb_abnormal.csv.zip, ptbdb_normal.csv.zip, mitbih_train.csv.zip, mitbih_test.csv.zip"
        )

    print(f"[EXTRACT] {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    csv_path = find_file_case_insensitive(raw_dir, csv_name)
    if csv_path is None:
        all_csv = sorted(raw_dir.rglob("*.csv"))
        raise FileNotFoundError(
            f"Could not find {csv_name} after extracting {zip_path}. Found CSV files: {all_csv[:20]}"
        )

    return csv_path


def load_csv_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    arr = df.values.astype(np.float32)

    if arr.shape[1] < 2:
        raise ValueError(f"{path} has too few columns: {arr.shape}")

    x = arr[:, :-1].astype(np.float32)
    y = arr[:, -1].astype(int)

    return x, y


def maybe_subsample(x: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_n <= 0 or len(y) <= max_n:
        return x, y

    idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=max_n,
        stratify=y,
        random_state=seed,
    )
    return x[idx], y[idx]


def class_distribution(y: np.ndarray, label_map: dict[int, str]) -> list[dict]:
    rows = []
    total = len(y)

    for c in sorted(np.unique(y).astype(int).tolist()):
        n = int((y == c).sum())
        rows.append({
            "class_id": int(c),
            "class_name": label_map.get(int(c), str(c)),
            "count": n,
            "pct": float(n / total) if total else 0.0,
        })

    return rows


class ECGBeatDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch = out_ch // 4

        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, branch, kernel_size=1),
            nn.BatchNorm1d(branch),
            nn.ReLU(),
        )

        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch),
            nn.ReLU(),
        )

        self.b5 = nn.Sequential(
            nn.Conv1d(in_ch, branch, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch),
            nn.ReLU(),
        )

        self.bp = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch, kernel_size=1),
            nn.BatchNorm1d(branch),
            nn.ReLU(),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class Inception1D(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            InceptionBlock1D(32, 64),
            InceptionBlock1D(64, 128),
            InceptionBlock1D(128, 128),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)


def compute_torch_class_weights(y: np.ndarray, n_classes: int, device: str):
    classes = np.arange(n_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_torch(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    ys = []
    ps = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()

        ps.append(prob)
        ys.append(yb.numpy())

    return np.concatenate(ys), np.concatenate(ps)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[nn.Module, list[dict]]:
    model = model.to(device)

    train_loader = DataLoader(
        ECGBeatDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        ECGBeatDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    weights = compute_torch_class_weights(y_train, n_classes=n_classes, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_state = None
    patience = 4
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(yb)

        y_true, y_prob = predict_torch(model, val_loader, device)
        y_pred = y_prob.argmax(axis=1)

        val_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        val_bal_acc = balanced_accuracy_score(y_true, y_pred)
        train_loss = total_loss / max(1, len(y_train))

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_macro_f1": float(val_macro_f1),
            "val_balanced_accuracy": float(val_bal_acc),
        }
        history.append(row)

        print(
            f"[TORCH] epoch={epoch} "
            f"train_loss={train_loss:.5f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"val_bal_acc={val_bal_acc:.4f}"
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("[TORCH] early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ensure_dir(out_path.parent)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_classes": n_classes,
        "model_class": model.__class__.__name__,
        "history": history,
    }, out_path)

    return model, history


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_map: dict[int, str],
    out_dir: Path,
    prefix: str,
    model_name: str,
    latency_ms_per_sample: float | None = None,
) -> dict:
    ensure_dir(out_dir)

    n_classes = y_prob.shape[1]
    labels = list(range(n_classes))
    class_names = [label_map.get(i, str(i)) for i in labels]

    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "model": model_name,
        "prefix": prefix,
        "n_samples": int(len(y_true)),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "latency_ms_per_sample": clean_float(latency_ms_per_sample),
    }

    try:
        if n_classes == 2:
            metrics["auroc_macro"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics["auprc_macro"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=labels)
            metrics["auroc_macro"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
            metrics["auprc_macro"] = float(average_precision_score(y_bin, y_prob, average="macro"))
    except Exception as e:
        metrics["auroc_macro"] = None
        metrics["auprc_macro"] = None
        metrics["auroc_auprc_error"] = str(e)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        per_class.append({
            "class_id": int(i),
            "class_name": name,
            "support": int((y_true == i).sum()),
            "precision": float(report[name]["precision"]) if name in report else 0.0,
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(report[name]["f1-score"]) if name in report else 0.0,
        })

    metrics["per_class"] = per_class

    (out_dir / f"{prefix}_{model_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(per_class).to_csv(
        out_dir / f"{prefix}_{model_name}_per_class.csv",
        index=False,
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} {model_name} confusion matrix")
    plt.xticks(np.arange(n_classes), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(n_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_{model_name}_confusion_matrix.png", dpi=220)
    plt.close()

    return metrics


def flatten_leaderboard_row(
    task_name: str,
    model_name: str,
    metrics: dict,
    n_train: int,
    n_test: int,
) -> dict:
    return {
        "task": task_name,
        "model": model_name,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_sensitivity_recall": metrics.get("macro_sensitivity_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "auroc_macro": metrics.get("auroc_macro"),
        "auprc_macro": metrics.get("auprc_macro"),
        "latency_ms_per_sample": metrics.get("latency_ms_per_sample"),
    }


def train_eval_sklearn_models(
    task_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict[int, str],
    out_dir: Path,
    seed: int,
    mode: str,
) -> list[dict]:
    n_estimators = 60 if mode == "smoke" else 160 if mode == "quick" else 300

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    rows = []
    ensure_dir(out_dir / "models")

    for name, model in models.items():
        print(f"[SKLEARN] train {task_name} {name}")

        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        train_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_prob = model.predict_proba(x_test)
        infer_sec = time.perf_counter() - t1
        latency_ms = 1000.0 * infer_sec / max(1, len(y_test))

        metrics = evaluate_predictions(
            y_true=y_test,
            y_prob=y_prob,
            label_map=label_map,
            out_dir=out_dir / "metrics",
            prefix=task_name,
            model_name=name,
            latency_ms_per_sample=latency_ms,
        )
        metrics["train_seconds"] = float(train_sec)

        joblib.dump(model, out_dir / "models" / f"{task_name}_{name}.joblib")

        rows.append(
            flatten_leaderboard_row(
                task_name=task_name,
                model_name=name,
                metrics=metrics,
                n_train=len(y_train),
                n_test=len(y_test),
            )
        )

    return rows


def train_eval_deep_models(
    task_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_map: dict[int, str],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
) -> list[dict]:
    n_classes = len(label_map)
    rows = []

    specs = [
        ("cnn1d", CNN1D(n_classes=n_classes)),
        ("inception1d", Inception1D(n_classes=n_classes)),
    ]

    for name, model in specs:
        print(f"[DEEP] train {task_name} {name}")

        out_path = out_dir / "models" / f"{task_name}_{name}.pt"

        t0 = time.perf_counter()
        model, history = train_torch_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            n_classes=n_classes,
            out_path=out_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=1e-3,
            device=device,
        )
        train_sec = time.perf_counter() - t0

        test_loader = DataLoader(
            ECGBeatDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        t1 = time.perf_counter()
        y_true, y_prob = predict_torch(model, test_loader, device)
        infer_sec = time.perf_counter() - t1
        latency_ms = 1000.0 * infer_sec / max(1, len(y_true))

        metrics = evaluate_predictions(
            y_true=y_true,
            y_prob=y_prob,
            label_map=label_map,
            out_dir=out_dir / "metrics",
            prefix=task_name,
            model_name=name,
            latency_ms_per_sample=latency_ms,
        )
        metrics["train_seconds"] = float(train_sec)
        metrics["history"] = history

        (out_dir / "metrics" / f"{task_name}_{name}_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        rows.append(
            flatten_leaderboard_row(
                task_name=task_name,
                model_name=name,
                metrics=metrics,
                n_train=len(y_train),
                n_test=len(y_test),
            )
        )

    return rows


def prepare_datasets(raw_dir: Path, out_dir: Path, seed: int, mode: str) -> dict:
    ensure_dir(raw_dir)

    mit_train_path = extract_zip_if_needed(raw_dir, "mitbih_train.csv.zip", "mitbih_train.csv")
    mit_test_path = extract_zip_if_needed(raw_dir, "mitbih_test.csv.zip", "mitbih_test.csv")
    ptb_normal_path = extract_zip_if_needed(raw_dir, "ptbdb_normal.csv.zip", "ptbdb_normal.csv")
    ptb_abnormal_path = extract_zip_if_needed(raw_dir, "ptbdb_abnormal.csv.zip", "ptbdb_abnormal.csv")

    x_mit_train, y_mit_train = load_csv_matrix(mit_train_path)
    x_mit_test, y_mit_test = load_csv_matrix(mit_test_path)

    x_ptb_normal, _ = load_csv_matrix(ptb_normal_path)
    x_ptb_abnormal, _ = load_csv_matrix(ptb_abnormal_path)

    y_ptb_normal = np.zeros(len(x_ptb_normal), dtype=int)
    y_ptb_abnormal = np.ones(len(x_ptb_abnormal), dtype=int)

    x_ptb = np.concatenate([x_ptb_normal, x_ptb_abnormal], axis=0)
    y_ptb = np.concatenate([y_ptb_normal, y_ptb_abnormal], axis=0)

    if mode == "smoke":
        x_mit_train, y_mit_train = maybe_subsample(x_mit_train, y_mit_train, 6000, seed)
        x_mit_test, y_mit_test = maybe_subsample(x_mit_test, y_mit_test, 2000, seed)
        x_ptb, y_ptb = maybe_subsample(x_ptb, y_ptb, 4000, seed)
    elif mode == "quick":
        x_mit_train, y_mit_train = maybe_subsample(x_mit_train, y_mit_train, 30000, seed)
        x_mit_test, y_mit_test = maybe_subsample(x_mit_test, y_mit_test, 8000, seed)
        x_ptb, y_ptb = maybe_subsample(x_ptb, y_ptb, 12000, seed)

    x_mit_train2, x_mit_val, y_mit_train2, y_mit_val = train_test_split(
        x_mit_train,
        y_mit_train,
        test_size=0.15,
        stratify=y_mit_train,
        random_state=seed,
    )

    x_ptb_train_all, x_ptb_test, y_ptb_train_all, y_ptb_test = train_test_split(
        x_ptb,
        y_ptb,
        test_size=0.20,
        stratify=y_ptb,
        random_state=seed,
    )

    x_ptb_train, x_ptb_val, y_ptb_train, y_ptb_val = train_test_split(
        x_ptb_train_all,
        y_ptb_train_all,
        test_size=0.15,
        stratify=y_ptb_train_all,
        random_state=seed,
    )

    summary = {
        "version": "beatscope_v28",
        "mode": mode,
        "raw_dir": str(raw_dir),
        "signal_points_per_beat": int(x_mit_train.shape[1]),
        "sampling_frequency_hz": 125,
        "mitbih": {
            "train_shape": list(x_mit_train.shape),
            "test_shape": list(x_mit_test.shape),
            "label_map": MITBIH_LABELS,
            "train_distribution": class_distribution(y_mit_train, MITBIH_LABELS),
            "test_distribution": class_distribution(y_mit_test, MITBIH_LABELS),
        },
        "ptbdb": {
            "total_shape": list(x_ptb.shape),
            "label_map": PTBDB_LABELS,
            "distribution": class_distribution(y_ptb, PTBDB_LABELS),
            "split": {
                "train": int(len(y_ptb_train)),
                "val": int(len(y_ptb_val)),
                "test": int(len(y_ptb_test)),
            },
        },
        "claim_boundary": (
            "BeatScope v2.8 is a beat-level auxiliary benchmark using segmented/preprocessed heartbeat vectors. "
            "It is not mixed with CardioTwin-AI v2.7 12-lead record-level validation."
        ),
    }

    (out_dir / "heartbeat_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(summary["mitbih"]["train_distribution"]).to_csv(
        out_dir / "mitbih_class_distribution_train.csv",
        index=False,
    )

    pd.DataFrame(summary["mitbih"]["test_distribution"]).to_csv(
        out_dir / "mitbih_class_distribution_test.csv",
        index=False,
    )

    pd.DataFrame(summary["ptbdb"]["distribution"]).to_csv(
        out_dir / "ptbdb_class_distribution.csv",
        index=False,
    )

    return {
        "summary": summary,
        "mitbih": (x_mit_train2, y_mit_train2, x_mit_val, y_mit_val, x_mit_test, y_mit_test),
        "ptbdb": (x_ptb_train, y_ptb_train, x_ptb_val, y_ptb_val, x_ptb_test, y_ptb_test),
    }


def run_transfer_learning(
    x_mit_train: np.ndarray,
    y_mit_train: np.ndarray,
    x_mit_val: np.ndarray,
    y_mit_val: np.ndarray,
    x_ptb_train: np.ndarray,
    y_ptb_train: np.ndarray,
    x_ptb_val: np.ndarray,
    y_ptb_val: np.ndarray,
    x_ptb_test: np.ndarray,
    y_ptb_test: np.ndarray,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
) -> tuple[pd.DataFrame, dict]:
    print("[TRANSFER] MIT-BIH pretrain -> PTBDB fine-tune")

    transfer_dir = out_dir / "transfer"
    ensure_dir(transfer_dir)
    ensure_dir(transfer_dir / "models")
    ensure_dir(transfer_dir / "metrics")

    pretrain = Inception1D(n_classes=5)
    pretrain, _ = train_torch_model(
        model=pretrain,
        x_train=x_mit_train,
        y_train=y_mit_train,
        x_val=x_mit_val,
        y_val=y_mit_val,
        n_classes=5,
        out_path=transfer_dir / "models" / "mitbih_pretrained_inception1d_encoder.pt",
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        device=device,
    )

    scratch = Inception1D(n_classes=2)
    scratch, _ = train_torch_model(
        model=scratch,
        x_train=x_ptb_train,
        y_train=y_ptb_train,
        x_val=x_ptb_val,
        y_val=y_ptb_val,
        n_classes=2,
        out_path=transfer_dir / "models" / "ptbdb_inception1d_scratch.pt",
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        device=device,
    )

    transfer = Inception1D(n_classes=2)

    source_state = pretrain.state_dict()
    target_state = transfer.state_dict()

    compatible = {
        k: v
        for k, v in source_state.items()
        if k in target_state and target_state[k].shape == v.shape and not k.startswith("head.")
    }

    target_state.update(compatible)
    transfer.load_state_dict(target_state)

    transfer, _ = train_torch_model(
        model=transfer,
        x_train=x_ptb_train,
        y_train=y_ptb_train,
        x_val=x_ptb_val,
        y_val=y_ptb_val,
        n_classes=2,
        out_path=transfer_dir / "models" / "ptbdb_inception1d_transfer.pt",
        epochs=epochs,
        batch_size=batch_size,
        lr=7e-4,
        device=device,
    )

    test_loader = DataLoader(
        ECGBeatDataset(x_ptb_test, y_ptb_test),
        batch_size=batch_size,
        shuffle=False,
    )

    rows = []

    for name, model in [
        ("ptbdb_inception1d_scratch", scratch),
        ("ptbdb_inception1d_transfer", transfer),
    ]:
        t0 = time.perf_counter()
        y_true, y_prob = predict_torch(model, test_loader, device)
        infer_sec = time.perf_counter() - t0
        latency_ms = 1000.0 * infer_sec / max(1, len(y_true))

        metrics = evaluate_predictions(
            y_true=y_true,
            y_prob=y_prob,
            label_map=PTBDB_LABELS,
            out_dir=transfer_dir / "metrics",
            prefix="transfer_ptbdb",
            model_name=name,
            latency_ms_per_sample=latency_ms,
        )

        rows.append(
            flatten_leaderboard_row(
                task_name="transfer_ptbdb",
                model_name=name,
                metrics=metrics,
                n_train=len(y_ptb_train),
                n_test=len(y_ptb_test),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(transfer_dir / "transfer_learning_leaderboard.csv", index=False)
    safe_to_markdown(df, transfer_dir / "transfer_learning_leaderboard.md")

    scratch_row = df[df["model"] == "ptbdb_inception1d_scratch"].iloc[0].to_dict()
    transfer_row = df[df["model"] == "ptbdb_inception1d_transfer"].iloc[0].to_dict()

    gains = {}
    for metric in ["balanced_accuracy", "macro_f1", "auroc_macro", "auprc_macro"]:
        a = clean_float(scratch_row.get(metric))
        b = clean_float(transfer_row.get(metric))
        gains[metric + "_gain"] = None if a is None or b is None else b - a

    summary = {
        "task": "MIT-BIH pretrain -> PTBDB fine-tune",
        "scratch": scratch_row,
        "transfer": transfer_row,
        "gains": gains,
        "interpretation": (
            "Positive gains indicate that MIT-BIH beat morphology pretraining improves PTBDB binary beat classification. "
            "Small or negative gains should be reported honestly as possible task/domain mismatch."
        ),
    }

    (transfer_dir / "transfer_learning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return df, summary


def build_html_report(
    out_dir: Path,
    dataset_summary: dict,
    mitbih_lb: pd.DataFrame,
    ptbdb_lb: pd.DataFrame,
    transfer_summary: dict,
) -> None:
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CardioTwin-AI BeatScope v2.8 Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 36px; line-height: 1.45; }}
h1, h2 {{ color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 7px; }}
th {{ background: #f3f4f6; }}
.warning {{ padding: 12px; background: #fff7ed; border-left: 4px solid #f97316; }}
pre {{ background: #f8fafc; padding: 12px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>CardioTwin-AI v2.8 BeatScope Benchmark Pack</h1>

<div class="warning">
<strong>Claim boundary:</strong>
BeatScope is a beat-level auxiliary benchmark. It is not mixed with the 12-lead record-level CardioTwin-AI v2.7 validation.
</div>

<h2>Dataset Summary</h2>
<pre>{json.dumps(dataset_summary, indent=2, ensure_ascii=False)}</pre>

<h2>MIT-BIH Beat-level Leaderboard</h2>
{mitbih_lb.to_html(index=False)}

<h2>PTBDB Binary Leaderboard</h2>
{ptbdb_lb.to_html(index=False)}

<h2>Transfer Learning Summary</h2>
<pre>{json.dumps(transfer_summary, indent=2, ensure_ascii=False)}</pre>

<h2>Interpretation</h2>
<p>
BeatScope v2.8 evaluates beat-level morphology classification and transfer learning using segmented ECG heartbeat vectors.
The results should be reported separately from 12-lead record-level validation because the input unit is a preprocessed beat segment rather than a full 12-lead ECG record.
</p>
</body>
</html>
"""

    (out_dir / "transfer_learning_report.html").write_text(html, encoding="utf-8")


def write_markdown_summary(
    out_dir: Path,
    dataset_summary: dict,
    mitbih_lb: pd.DataFrame,
    ptbdb_lb: pd.DataFrame,
    transfer_summary: dict,
) -> None:
    best_mit = mitbih_lb.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")
    best_ptb = ptbdb_lb.sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records")

    text = f"""# CardioTwin-AI v2.8 BeatScope Benchmark Summary

## Purpose

BeatScope v2.8 adds an auxiliary beat-level ECG benchmark branch to CardioTwin-AI.

This branch uses segmented, preprocessed heartbeat vectors from the Kaggle ECG Heartbeat Categorization Dataset. It is separate from the frozen CardioTwin-AI v2.7 12-lead record-level validation.

## Dataset Summary

- MIT-BIH train shape: `{dataset_summary["mitbih"]["train_shape"]}`
- MIT-BIH test shape: `{dataset_summary["mitbih"]["test_shape"]}`
- PTBDB total shape: `{dataset_summary["ptbdb"]["total_shape"]}`
- Signal points per beat: `{dataset_summary["signal_points_per_beat"]}`
- Sampling frequency: `{dataset_summary["sampling_frequency_hz"]} Hz`

## MIT-BIH Models

{mitbih_lb.to_string(index=False)}

Best MIT-BIH model by Macro-F1:

{json.dumps(best_mit, indent=2, ensure_ascii=False)}

## PTBDB Models

{ptbdb_lb.to_string(index=False)}

Best PTBDB model by Macro-F1:

{json.dumps(best_ptb, indent=2, ensure_ascii=False)}

## Transfer Learning

{json.dumps(transfer_summary, indent=2, ensure_ascii=False)}

## Claim Boundary

BeatScope is an auxiliary beat-level benchmark. It should not be mixed with the 12-lead record-level metrics from CardioTwin-AI v2.7.

Recommended wording:

BeatScope v2.8 evaluates beat-level morphology classification and transfer learning as an auxiliary benchmark, while CardioTwin-AI v2.7 remains the frozen 12-lead record-level safety-calibrated digital twin release.

## Outputs

- heartbeat_dataset_summary.json
- mitbih_model_leaderboard.csv
- ptbdb_model_leaderboard.csv
- transfer_learning_report.html
- beat_benchmark_summary.md
- metrics/
- models/
- transfer/
"""

    (out_dir / "beat_benchmark_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw-dir", default="data/raw/kaggle_heartbeat")
    ap.add_argument("--out-dir", default="artifacts/heartbeat_benchmark_v28")
    ap.add_argument("--mode", choices=["smoke", "quick", "full"], default="quick")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "models")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] BeatScope v2.8")
    print(f"[INFO] device={device}")
    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] raw_dir={raw_dir}")
    print(f"[INFO] out_dir={out_dir}")

    data = prepare_datasets(
        raw_dir=raw_dir,
        out_dir=out_dir,
        seed=args.seed,
        mode=args.mode,
    )

    summary = data["summary"]

    x_mit_train, y_mit_train, x_mit_val, y_mit_val, x_mit_test, y_mit_test = data["mitbih"]
    x_ptb_train, y_ptb_train, x_ptb_val, y_ptb_val, x_ptb_test, y_ptb_test = data["ptbdb"]

    mitbih_rows = train_eval_sklearn_models(
        task_name="mitbih",
        x_train=np.concatenate([x_mit_train, x_mit_val], axis=0),
        y_train=np.concatenate([y_mit_train, y_mit_val], axis=0),
        x_test=x_mit_test,
        y_test=y_mit_test,
        label_map=MITBIH_LABELS,
        out_dir=out_dir,
        seed=args.seed,
        mode=args.mode,
    )

    ptbdb_rows = train_eval_sklearn_models(
        task_name="ptbdb",
        x_train=np.concatenate([x_ptb_train, x_ptb_val], axis=0),
        y_train=np.concatenate([y_ptb_train, y_ptb_val], axis=0),
        x_test=x_ptb_test,
        y_test=y_ptb_test,
        label_map=PTBDB_LABELS,
        out_dir=out_dir,
        seed=args.seed,
        mode=args.mode,
    )

    if not args.skip_deep:
        mitbih_rows += train_eval_deep_models(
            task_name="mitbih",
            x_train=x_mit_train,
            y_train=y_mit_train,
            x_val=x_mit_val,
            y_val=y_mit_val,
            x_test=x_mit_test,
            y_test=y_mit_test,
            label_map=MITBIH_LABELS,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

        ptbdb_rows += train_eval_deep_models(
            task_name="ptbdb",
            x_train=x_ptb_train,
            y_train=y_ptb_train,
            x_val=x_ptb_val,
            y_val=y_ptb_val,
            x_test=x_ptb_test,
            y_test=y_ptb_test,
            label_map=PTBDB_LABELS,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

    mitbih_lb = pd.DataFrame(mitbih_rows).sort_values("macro_f1", ascending=False)
    ptbdb_lb = pd.DataFrame(ptbdb_rows).sort_values("macro_f1", ascending=False)

    mitbih_lb.to_csv(out_dir / "mitbih_model_leaderboard.csv", index=False)
    ptbdb_lb.to_csv(out_dir / "ptbdb_model_leaderboard.csv", index=False)

    safe_to_markdown(mitbih_lb, out_dir / "mitbih_model_leaderboard.md")
    safe_to_markdown(ptbdb_lb, out_dir / "ptbdb_model_leaderboard.md")

    if not args.skip_deep and not args.skip_transfer:
        transfer_lb, transfer_summary = run_transfer_learning(
            x_mit_train=x_mit_train,
            y_mit_train=y_mit_train,
            x_mit_val=x_mit_val,
            y_mit_val=y_mit_val,
            x_ptb_train=x_ptb_train,
            y_ptb_train=y_ptb_train,
            x_ptb_val=x_ptb_val,
            y_ptb_val=y_ptb_val,
            x_ptb_test=x_ptb_test,
            y_ptb_test=y_ptb_test,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        transfer_summary = {
            "status": "skipped",
            "reason": "skip_deep or skip_transfer was set",
        }

        (out_dir / "transfer_learning_report.html").write_text(
            "<html><body><h1>Transfer Learning Skipped</h1></body></html>",
            encoding="utf-8",
        )

    build_html_report(
        out_dir=out_dir,
        dataset_summary=summary,
        mitbih_lb=mitbih_lb,
        ptbdb_lb=ptbdb_lb,
        transfer_summary=transfer_summary,
    )

    write_markdown_summary(
        out_dir=out_dir,
        dataset_summary=summary,
        mitbih_lb=mitbih_lb,
        ptbdb_lb=ptbdb_lb,
        transfer_summary=transfer_summary,
    )

    final = {
        "version": "beatscope_v28",
        "mode": args.mode,
        "device": device,
        "outputs": {
            "dataset_summary": str(out_dir / "heartbeat_dataset_summary.json"),
            "mitbih_leaderboard": str(out_dir / "mitbih_model_leaderboard.csv"),
            "ptbdb_leaderboard": str(out_dir / "ptbdb_model_leaderboard.csv"),
            "transfer_report": str(out_dir / "transfer_learning_report.html"),
            "summary_md": str(out_dir / "beat_benchmark_summary.md"),
        },
        "best_mitbih_by_macro_f1": mitbih_lb.head(1).to_dict(orient="records"),
        "best_ptbdb_by_macro_f1": ptbdb_lb.head(1).to_dict(orient="records"),
        "transfer_summary": transfer_summary,
        "claim_boundary": "Beat-level auxiliary benchmark; do not mix with 12-lead record-level validation metrics.",
    }

    (out_dir / "beatscope_v28_run_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== BeatScope v2.8 DONE ===")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
