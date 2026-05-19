# matrix_calculator_app.py
# ------------------------------------------------------------
# Streamlit front-end for Matrix Calculator
# FIXED:
#  - No hard-coded GT column names (e.g. MORPHOLOGY)
#  - Uses resolve_gt_column everywhere
#  - Compatible with registry schemas (MORP, GRADE, LATER, etc.)
# UPDATED:
#  - Shows ALL records instead of only sample rows
#  - Adds GT original + GT mapped
#  - Adds Prediction original + Prediction mapped
#  - Forces Pred_mapped into the same canonical label space as GT_mapped
#  - Fixes incorrect morphology mapping (e.g. ductal vs lobular)
# ------------------------------------------------------------

import os
import io
import re
import zipfile
import difflib
import pandas as pd
import streamlit as st

from agents.matrix_calculator_agent import (
    CATEGORIES,
    merge_predictions_with_gt,
    filter_df_by_site,
    calculate_category_metrics,
    plot_confusion_matrix,
    resolve_gt_column,
)

# ============================== Config ==============================

OUT_REPORTS = "OUTPUTS/reports"
OUT_CM = "OUTPUTS/confusion_matrices"
os.makedirs(OUT_REPORTS, exist_ok=True)
os.makedirs(OUT_CM, exist_ok=True)

ALLOWED_SITES = {"BREAST", "COLORECTAL", "THYROID", "PROSTATE"}

st.set_page_config(
    page_title="Matrix Calculator — Category × Site",
    layout="wide"
)

st.title("Matrix Calculator — Evaluation (Category × Site)")

# ============================== Sidebar ==============================

st.sidebar.header("Inputs")
pred_file = st.sidebar.file_uploader("Predictions CSV", type=["csv"])
gt_file = st.sidebar.file_uploader("Ground Truth CSV", type=["csv"])
join_key = st.sidebar.text_input("Join key (e.g. mrn)", value="mrn")

run_btn = st.sidebar.button("Run evaluation")

# ============================== Helpers ==============================

def resolve_site_col(df: pd.DataFrame):
    for c in ("TOPOLOGY", "TOPO"):
        if c in df.columns:
            return c
    return None


def resolve_join_key(pred_df, gt_df, user_key):
    if user_key in pred_df.columns and user_key in gt_df.columns:
        return user_key

    candidates = ["mrn", "patient_id", "patientrecordid", "pers", "sourceno"]
    for c in candidates:
        if c in pred_df.columns and c in gt_df.columns:
            st.warning(f"Using '{c}' as join key (auto-detected).")
            return c

    raise ValueError(
        f"No common join key found.\n"
        f"Pred columns: {list(pred_df.columns)}\n"
        f"GT columns: {list(gt_df.columns)}"
    )


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_lower_str(x):
    return safe_str(x).lower()


def normalize_text(x):
    x = safe_str(x).lower()
    x = re.sub(r"[_\-\/]+", " ", x)
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def resolve_prediction_raw_col(site_df: pd.DataFrame, category: str):
    candidates = [
        f"{category}_response",
        f"{category}_raw",
        f"{category}_output",
        f"{category}_prediction",
        f"{category}_pred",
        f"{category}_estimated",
    ]
    for c in candidates:
        if c in site_df.columns:
            return c
    raise ValueError(f"No prediction column found for category '{category}'.")


def resolve_prediction_eval_col(site_df: pd.DataFrame, category: str, pred_raw_col: str):
    candidates = [
        f"{category}_estimated",
        f"{category}_mapped",
        f"{category}_normalized",
        f"{category}_cleaned",
        pred_raw_col,
    ]
    for c in candidates:
        if c in site_df.columns:
            return c
    return pred_raw_col


def resolve_gt_mapped_col(site_df: pd.DataFrame, category: str, gt_raw_col: str):
    candidates = [
        f"{category}_gt_mapped",
        f"{category}_mapped_gt",
        f"{category}_normalized_gt",
        f"{category}_cleaned_gt",
        gt_raw_col,
    ]
    for c in candidates:
        if c in site_df.columns:
            return c
    return gt_raw_col


def build_alias_map(valid_labels):
    """
    Build lightweight aliases from GT canonical labels.
    """
    alias_map = {}
    for label in valid_labels:
        norm = normalize_text(label)
        if norm:
            alias_map[norm] = label

        parts = re.split(r";|,|\bor\b|\balso called\b", norm)
        for p in parts:
            p = p.strip()
            if p:
                alias_map[p] = label

    return alias_map


def detect_unknown_label(valid_gt_labels):
    for lbl in valid_gt_labels:
        norm = normalize_text(lbl)
        if "unknown" in norm:
            return lbl
    return "unknown"


def extract_morph_key_tokens(text):
    """
    Extract morphology-defining tokens.
    These tokens must agree between prediction and GT candidate.
    """
    text = normalize_text(text)
    tokens = set(text.split())

    key_groups = {
        "ductal": {"duct", "ductal"},
        "lobular": {"lobular"},
        "mucinous": {"mucinous"},
        "papillary": {"papillary"},
        "acinar": {"acinar"},
        "phyllodes": {"phyllodes"},
        "apocrine": {"apocrine"},
        "follicular": {"follicular"},
        "anaplastic": {"anaplastic"},
        "micropapillary": {"micropapillary"},
        "medullary": {"medullary"},
        "serous": {"serous"},
        "endometrioid": {"endometrioid"},
        "clear": {"clear"},
        "squamous": {"squamous"},
    }

    found = set()
    for canonical, variants in key_groups.items():
        if tokens & variants:
            found.add(canonical)

    return found


def map_prediction_to_gt_label(pred_value, valid_gt_labels, category=None):
    """
    Force prediction into one of the canonical GT labels.
    Uses safer rules for morphology to avoid wrong mappings
    like ductal -> lobular.
    """
    pred_raw = safe_str(pred_value)
    unknown_label = detect_unknown_label(valid_gt_labels)

    if not pred_raw:
        return unknown_label

    if not valid_gt_labels:
        return pred_raw

    pred_norm = normalize_text(pred_raw)
    alias_map = build_alias_map(valid_gt_labels)

    # 1) exact normalized alias match
    if pred_norm in alias_map:
        return alias_map[pred_norm]

    # 2) morphology hard constraint
    if safe_str(category).lower() == "morphology":
        pred_keys = extract_morph_key_tokens(pred_raw)

        if pred_keys:
            filtered_candidates = []
            for lbl in valid_gt_labels:
                gt_keys = extract_morph_key_tokens(lbl)

                # if GT has morphology keys, require overlap
                if gt_keys:
                    if pred_keys & gt_keys:
                        filtered_candidates.append(lbl)
                else:
                    # if GT has no extracted keys, don't allow it when pred has strong keys
                    continue

            if filtered_candidates:
                # exact alias within filtered candidates
                filtered_alias_map = build_alias_map(filtered_candidates)
                if pred_norm in filtered_alias_map:
                    return filtered_alias_map[pred_norm]

                # substring match within filtered candidates
                for alias, canonical in filtered_alias_map.items():
                    if alias and (alias in pred_norm or pred_norm in alias):
                        return canonical

                # fuzzy match within filtered candidates
                best = difflib.get_close_matches(
                    pred_norm,
                    list(filtered_alias_map.keys()),
                    n=1,
                    cutoff=0.6
                )
                if best:
                    return filtered_alias_map[best[0]]

                # fallback: if nothing good found, keep original prediction text
                return pred_raw

            # if prediction has strong morphology keys but no valid GT overlap,
            # do not force to wrong subtype
            return pred_raw

    # 3) substring match for non-morphology or if no strong morphology keys
    for alias, canonical in alias_map.items():
        if alias and (alias in pred_norm or pred_norm in alias):
            return canonical

    # 4) fuzzy match
    best = difflib.get_close_matches(pred_norm, list(alias_map.keys()), n=1, cutoff=0.6)
    if best:
        return alias_map[best[0]]

    # 5) if nothing matched, keep original prediction text
    return pred_raw


# ============================== Run ==============================

if run_btn:
    if pred_file is None or gt_file is None:
        st.error("Please upload BOTH Predictions and Ground Truth CSV files.")
        st.stop()

    pred_df = pd.read_csv(pred_file)
    gt_df = pd.read_csv(gt_file)

    try:
        join_key_resolved = resolve_join_key(pred_df, gt_df, join_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

    merged_df = merge_predictions_with_gt(pred_df, gt_df, join_key_resolved)

    site_col = resolve_site_col(merged_df)
    if site_col is None:
        st.error("No site column found (expected TOPOLOGY or TOPO).")
        st.stop()

    merged_df[site_col] = merged_df[site_col].astype(str).str.upper().str.strip()
    merged_df = merged_df[merged_df[site_col].isin(ALLOWED_SITES)].copy()

    if merged_df.empty:
        st.warning("No rows remaining after site filtering.")
        st.stop()

    detected_sites = sorted(merged_df[site_col].unique().tolist())

    st.sidebar.header("Sites")
    chosen_sites = st.sidebar.multiselect(
        "Select sites",
        options=detected_sites,
        default=detected_sites,
    )

    if not chosen_sites:
        st.warning("Please select at least one site.")
        st.stop()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        category_tabs = st.tabs([c.upper() for c in CATEGORIES])

        for c_idx, category in enumerate(CATEGORIES):
            with category_tabs[c_idx]:
                st.header(category.upper())

                site_tabs = st.tabs(chosen_sites)

                for s_idx, site in enumerate(chosen_sites):
                    with site_tabs[s_idx]:
                        st.subheader(f"{category.upper()} — {site}")

                        # Hide Grade for PROSTATE only
                        if category.lower() == "grade" and site.upper() == "PROSTATE":
                            continue

                        site_df = filter_df_by_site(merged_df, site)
                        if site_df.empty:
                            st.warning("No records for this site.")
                            continue

                        # ---------- Evaluation ----------
                        try:
                            rep_df, cm, labels = calculate_category_metrics(
                                site_df, category
                            )
                        except Exception as e:
                            st.error(str(e))
                            continue

                        st.markdown("### Classification Report")
                        st.dataframe(rep_df, use_container_width=True)

                        # ---------- Full Records Table ----------
                        try:
                            gt_raw_col = resolve_gt_column(site_df, category)
                            gt_mapped_col = resolve_gt_mapped_col(site_df, category, gt_raw_col)

                            pred_raw_col = resolve_prediction_raw_col(site_df, category)
                            pred_eval_col = resolve_prediction_eval_col(site_df, category, pred_raw_col)

                            gt_original_series = site_df[gt_raw_col].apply(safe_str)
                            gt_mapped_series = site_df[gt_mapped_col].apply(safe_str)

                            pred_original_series = site_df[pred_raw_col].apply(safe_str)
                            pred_eval_series = site_df[pred_eval_col].apply(safe_str)

                            valid_gt_labels = sorted(
                                [x for x in gt_mapped_series.dropna().astype(str).str.strip().unique().tolist() if x]
                            )

                            pred_mapped_series = pred_eval_series.apply(
                                lambda x: map_prediction_to_gt_label(x, valid_gt_labels, category)
                            )

                            records_df = pd.DataFrame({
                                "mrn": site_df[join_key_resolved].astype(str).str.replace(",", "", regex=False),
                                "GT_original": gt_original_series,
                                "GT_mapped": gt_mapped_series,
                                "Pred_original": pred_original_series,
                                "Pred_mapped": pred_mapped_series,
                            })

                            records_df["match"] = (
                                records_df["GT_mapped"].apply(safe_lower_str)
                                ==
                                records_df["Pred_mapped"].apply(safe_lower_str)
                            )

                            st.markdown("### Full GT vs Predictions Records")
                            st.dataframe(records_df, use_container_width=True, height=500)

                            records_name = f"{category}_{site.lower()}_all_records.csv"
                            records_path = os.path.join(OUT_REPORTS, records_name)
                            records_df.to_csv(records_path, index=False)

                            with open(records_path, "rb") as f:
                                st.download_button(
                                    f"Download full records ({category}/{site})",
                                    f,
                                    file_name=records_name,
                                )
                            zf.write(records_path, f"reports/{records_name}")

                        except Exception as e:
                            st.warning(f"Could not build full records table: {e}")

                        # ---------- Save report ----------
                        report_name = f"{category}_{site.lower()}_classification_report.csv"
                        report_path = os.path.join(OUT_REPORTS, report_name)
                        rep_df.to_csv(report_path)

                        with open(report_path, "rb") as f:
                            st.download_button(
                                f"Download report ({category}/{site})",
                                f,
                                file_name=report_name,
                            )
                        zf.write(report_path, f"reports/{report_name}")

                        # ---------- Confusion Matrix ----------
                        st.markdown("### Confusion Matrix")
                        fig = plot_confusion_matrix(
                            cm,
                            labels,
                            title=f"{category.upper()} — {site}"
                        )
                        st.pyplot(fig)

                        cm_csv = f"{category}_{site.lower()}_confusion_matrix.csv"
                        cm_png = f"{category}_{site.lower()}_confusion_matrix.png"

                        cm_csv_path = os.path.join(OUT_CM, cm_csv)
                        cm_png_path = os.path.join(OUT_CM, cm_png)

                        pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_csv_path)
                        fig.savefig(cm_png_path, bbox_inches="tight")

                        for p in (cm_csv_path, cm_png_path):
                            with open(p, "rb") as f:
                                st.download_button(
                                    f"Download {os.path.basename(p)}",
                                    f,
                                    file_name=os.path.basename(p),
                                )
                            zf.write(p, f"confusion_matrices/{os.path.basename(p)}")

        st.markdown("---")
        zip_buffer.seek(0)
        st.download_button(
            "Download ALL outputs (ZIP)",
            zip_buffer,
            file_name="evaluation_outputs.zip",
            mime="application/zip",
        )

    st.success("Evaluation completed successfully.")