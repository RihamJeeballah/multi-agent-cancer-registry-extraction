from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mcp.middleware import log_message
import json
import re
import pandas as pd

# ---------------- Helpers ----------------

_MISSING_LIKE = {"", "none", "null", "na", "n/a", "nan", "unknown", "-", "--"}
def _is_missing_like(v) -> bool:
    if v is None: return True
    try: s = str(v).strip().lower()
    except Exception: return False
    return s in _MISSING_LIKE

def build_grade_closed_set_for_site(gt_df: pd.DataFrame, site: str) -> list[str]:
    """GT uniques for GRADE when TOPOLOGY/TOPO equals site (case-insensitive)."""
    site_col = "TOPOLOGY" if "TOPOLOGY" in gt_df.columns else ("TOPO" if "TOPO" in gt_df.columns else None)
    if site_col is None:
        raise ValueError("GT missing site column (TOPOLOGY/TOPO).")
    mask = gt_df[site_col].astype(str).str.upper().str.strip() == str(site).upper().strip()
    if "GRADE" in gt_df.columns:
        col = "GRADE"
    elif "grade" in gt_df.columns:
        col = "grade"
    else:
        raise ValueError("GT missing GRADE column.")
    vals = [str(v).strip() for v in gt_df.loc[mask, col].dropna().tolist() if str(v).strip()]
    labels = sorted(set(vals))
    # keep 'unknown' only if present in GT; you can force-add it if you prefer
    if "unknown" not in [v.lower() for v in labels]:
        labels.append("unknown")  # <- comment this line if you want it forbidden when absent in GT
    return labels

def llm_numbered_closed_set_map(llm: OllamaLLM, category: str, raw_value, valid_labels: list[str]) -> str:
    """Force LLM to choose nearest label from numbered list. Missing-like -> 'unknown'."""
    if _is_missing_like(raw_value):
        return "unknown" if any(v.lower() == "unknown" for v in valid_labels) else valid_labels[0]

    numbered = "\n".join(f"{i+1}) {lab}" for i, lab in enumerate(valid_labels))
    unknown_allowed = any(v.lower() == "unknown" for v in valid_labels)
    prompt = (
        "You are a medical coding normalizer.\n"
        f"Category: {category}\n\n"
        "Select the best match for the raw prediction from the options below.\n"
        "OPTIONS (answer with the number only):\n"
        f"{numbered}\n\n"
        f'RAW PREDICTION: "{str(raw_value)}"\n\n'
        "Rules:\n"
        "- Output ONLY the number (e.g., 2).\n"
        "- Match semantics (synonyms, abbreviations, roman numerals, punctuation, casing).\n"
        + ("- If unsure, choose the option labeled 'unknown'.\n" if unknown_allowed else "")
    )
    def _pick(ans: str) -> str | None:
        m = re.search(r"\d+", ans or "")
        if not m: return None
        idx = int(m.group(0)) - 1
        return valid_labels[idx] if 0 <= idx < len(valid_labels) else None

    try:
        ans = (llm.invoke(prompt) or "").strip()
    except Exception:
        ans = ""
    mapped = _pick(ans)
    if mapped: return mapped

    try:
        ans2 = (llm.invoke(prompt + "\nIMPORTANT: Reply with ONLY the number.") or "").strip()
    except Exception:
        ans2 = ""
    return _pick(ans2) or ("unknown" if unknown_allowed else valid_labels[0])

# ---------------- Dynamic extractor prompt ----------------

extract_prompt = PromptTemplate.from_template(
    """
You are an expert medical assistant specialized in cancer pathology information extraction.

Task: extract the tumor **grade** from the excerpt.

Return **JSON only** with four keys:
1. "grade stated": <exact grade phrase found in text, or "Unknown">,
2. "grade estimated": <your best estimate, but it MUST be one of the VALID VALUES below>,
3. "grade_CD": <confidence 0.00..1.00>,
4. "grade_evidence": <short quote/explanation from the text>.

VALID VALUES (choose exactly one verbatim for "grade estimated"):
{valid_values_bulleted}

If nothing supports an estimate, use "Unknown" for both stated and estimated.

Input:
{context}

Output (JSON only):
{{
  "grade stated": "...",
  "grade estimated": "...",
  "grade_CD": 0.0,
  "grade_evidence": "..."
}}
"""
)

# Compose the chain
def make_grade_chain(llm: OllamaLLM):
    return extract_prompt | llm | StrOutputParser()

# ---------------- Orchestrator ----------------

def grade_extractor(state, site: str, gt_df: pd.DataFrame, llm: OllamaLLM | None = None):
    """
    - Builds site-specific closed set from GT (GRADE uniques for the given site).
    - Runs the extractor with those values injected.
    - Maps the 'grade estimated' to that closed set again (numbered mapping) as a safety net.
    - Stores raw JSON, parsed fields, and mapped result into state.
    """
    if llm is None:
        llm = OllamaLLM(model="llama3.3:latest")

    # 1) Build closed set for the site
    valid_values = build_grade_closed_set_for_site(gt_df, site)
    bullets = "\n".join(f"- {v}" for v in valid_values)

    # 2) Build context from retrieved chunks
    context = "\n".join(state.get("retrieved_chunks", {}).get("grade", []))

    # 3) Run extractor with dynamic valid list
    chain = make_grade_chain(llm)
    raw_out = chain.invoke({"context": context, "valid_values_bulleted": bullets}).strip()

    # 4) Parse JSON (robustly)
    try:
        parsed = json.loads(raw_out)
    except Exception:
        # Try to salvage with common tricks
        try:
            raw_out = raw_out[raw_out.find("{"): raw_out.rfind("}") + 1]
            parsed = json.loads(raw_out)
        except Exception:
            parsed = {
                "grade stated": "Unknown",
                "grade estimated": "Unknown",
                "grade_CD": 0.0,
                "grade_evidence": ""
            }

    # 5) Safety-map the estimated label into the closed set (handles e.g. 'Grade III' vs 'group 3', etc.)
    est_raw = parsed.get("grade estimated", "Unknown")
    est_mapped = llm_numbered_closed_set_map(llm, "grade", est_raw, valid_values)

    # 6) Save into state
    state.setdefault("agent_outputs", {})["grade_extractor_raw_json"] = raw_out
    state["grade_stated"] = parsed.get("grade stated", "Unknown")
    state["grade_estimated_raw"] = est_raw
    state["grade_estimated_mapped"] = est_mapped
    state["grade_CD"] = parsed.get("grade_CD", 0.0)
    state["grade_evidence"] = parsed.get("grade_evidence", "")

    # Optional logging
    log_message("grade_extractor", "review_agent", json.dumps({
        "valid_values": valid_values,
        "extracted": parsed,
        "mapped_estimate": est_mapped
    }))

    return state
