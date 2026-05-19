# agents/matrix_calculator_agent.py
# ------------------------------------------------------------
# Deterministic evaluation for agentic medical IE
# FIXED:
#  - GT column alias resolution (incl. MORP, GRADE, LATER, TRE-1)
#  - Correct grade normalization (G3 → G2 → G1)
#  - No LLM usage in evaluation
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Tuple
import re
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================== Constants ==============================

CATEGORIES = ["grade", "morphology", "t", "n", "m", "laterality"]
SITE_COLS = ["TOPOLOGY", "TOPO"]

# ============================== GT Column Aliases ==============================

GT_ALIASES = {
    "grade": [
        "GRADE", "grade", "grade (desc)", "grade_gold"
    ],
    "morphology": [
        "MORP", "morp", "morp (desc)", "MORPHOLOGY", "morphology_gold"
    ],
    "laterality": [
        "LATER", "later", "later (desc)", "laterality_gold"
    ],
    "t": ["T", "t"],
    "n": ["N", "n"],
    "m": ["M", "m"],
}

def resolve_gt_column(df: pd.DataFrame, category: str) -> str:
    for col in GT_ALIASES.get(category, []):
        if col in df.columns:
            return col
    raise KeyError(
        f"No GT column found for category '{category}'. "
        f"Available columns: {list(df.columns)}"
    )

# ============================== Normalizers ==============================

def normalize_grade(v):
    if v is None:
        return "unknown"
    s = str(v).lower()

    # descending severity (IMPORTANT)
    if "grade iii" in s or "grade 3" in s or "poor" in s:
        return "G3"
    if "grade ii" in s or "grade 2" in s or "moderate" in s:
        return "G2"
    if "grade i" in s or "grade 1" in s or "well" in s:
        return "G1"

    return "unknown"


def normalize_tnm(v, axis):
    if v is None:
        return "unknown"

    s = str(v).lower()
    s = re.sub(r"[^a-z0-9]", "", s)

    if s in {"", "unknown", "na", "nan"}:
        return "unknown"

    if axis == "t":
        for x in ["tis", "t0", "t1", "t2", "t3", "t4"]:
            if x in s:
                return x.upper()

    if axis == "n":
        for x in ["n0", "n1", "n2", "n3"]:
            if x in s:
                return x.upper()

    if axis == "m":
        for x in ["m0", "m1"]:
            if x in s:
                return x.upper()

    if s.endswith("x"):
        return axis.upper() + "X"

    return "unknown"


def normalize_laterality(v):
    if v is None:
        return "unknown"

    s = str(v).lower()

    if "left" in s or "lt" in s:
        return "left"
    if "right" in s or "rt" in s:
        return "right"
    if "bilat" in s:
        return "bilateral"

    return "unknown"


def normalize_morphology(v):
    if v is None:
        return "unknown"

    s = str(v).lower()

    if "duct" in s:
        return "IDC"
    if "lobular" in s:
        return "ILC"
    if "adenocarcinoma" in s:
        return "Adenocarcinoma"
    if "mucin" in s:
        return "Mucinous"
    if "carcinoma" in s:
        return "Carcinoma"

    return "unknown"

# ============================== Utilities ==============================

def resolve_site_col(df: pd.DataFrame) -> str:
    for c in SITE_COLS:
        if c in df.columns:
            return c
    raise ValueError("No site column found (TOPOLOGY / TOPO).")


def filter_df_by_site(df: pd.DataFrame, site: str) -> pd.DataFrame:
    site_col = resolve_site_col(df)
    mask = df[site_col].astype(str).str.upper().str.contains(site.upper())
    return df[mask].copy()


def merge_predictions_with_gt(pred_df, gt_df, key: str):
    return pred_df.merge(gt_df, on=key, how="left", suffixes=("_pred", ""))

# ============================== Core Evaluation ==============================

def calculate_category_metrics(
    df: pd.DataFrame,
    category: str
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:

    if category == "grade":
        gt_col = resolve_gt_column(df, "grade")
        y_true = [normalize_grade(v) for v in df[gt_col]]
        y_pred = [normalize_grade(v) for v in df["grade_estimated"]]
        labels = ["G1", "G2", "G3", "unknown"]

    elif category == "morphology":
        gt_col = resolve_gt_column(df, "morphology")
        y_true = [normalize_morphology(v) for v in df[gt_col]]
        y_pred = [normalize_morphology(v) for v in df["morphology_estimated"]]
        labels = sorted(set(y_true + y_pred))

    elif category in {"t", "n", "m"}:
        gt_col = resolve_gt_column(df, category)
        y_true = [normalize_tnm(v, category) for v in df[gt_col]]
        y_pred = [normalize_tnm(v, category) for v in df[f"{category}_estimated"]]
        labels = sorted(set(y_true + y_pred))

    elif category == "laterality":
        gt_col = resolve_gt_column(df, "laterality")
        y_true = [normalize_laterality(v) for v in df[gt_col]]
        y_pred = [normalize_laterality(v) for v in df["laterality_estimated"]]
        labels = ["left", "right", "bilateral", "unknown"]

    else:
        raise ValueError(f"Unsupported category: {category}")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(report).transpose(), cm, labels

# ============================== Plotting ==============================

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    return fig
