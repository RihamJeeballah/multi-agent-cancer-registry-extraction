# /home/riham/Desktop/llm_based_info_extraction/Agentic_medical_ie/agents/reviewer_agent_8.py
#acts as a clinical quality-control and normalization layer that validates, 
# corrects, and standardizes the outputs of upstream extraction agents before final aggregation or storage.
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from mcp.middleware import log_message
from typing import Dict, Any
import json, re

llm = OllamaLLM(model="llama3.3:latest", temperature=0.0)

# ---------- helpers ----------
def _strip_invisibles(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\u200b", "").replace("\ufeff", "")

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s

def _extract_latest_json(text: str) -> dict:
    """Extract the last JSON object from arbitrary text; return {} if none."""
    if not text:
        return {}
    text = _strip_invisibles(text).strip()

    # 1) try whole
    try:
        obj = json.loads(_strip_fences(text))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) fenced JSON blocks
    blocks = re.findall(r"```(?:json)?\s*({[\s\S]*?})\s*```", text, flags=re.IGNORECASE)
    for raw in reversed(blocks):
        try:
            obj = json.loads(_strip_invisibles(raw).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # 3) any brace object
    objects = re.findall(r"({[\s\S]*?})", text)
    for raw in reversed(objects):
        try:
            obj = json.loads(_strip_invisibles(raw).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return {}

def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in ("true", "yes", "y", "1"):  return True
        if xs in ("false", "no", "n", "0"):  return False
    return False

def _sanitize_review(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist with correct types."""
    # guard obj
    if not isinstance(obj, dict):
        obj = {}
    out = {
        "grade_valid": _coerce_bool(obj.get("grade_valid")),
        "morphology_valid": _coerce_bool(obj.get("morphology_valid")),
        "morphology_mapped": obj.get("morphology_mapped"),
        "tnm_valid": _coerce_bool(obj.get("tnm_valid")),
        "laterality_valid": None if obj.get("laterality_valid") in (None, "null") else _coerce_bool(obj.get("laterality_valid")),
        "treatment_valid": None if obj.get("treatment_valid") in (None, "null") else _coerce_bool(obj.get("treatment_valid")),
        "comment": obj.get("comment") or "",
    }
    if out["morphology_mapped"] is not None and not isinstance(out["morphology_mapped"], str):
        out["morphology_mapped"] = str(out["morphology_mapped"])
    return out

def _stub_review() -> Dict[str, Any]:
    return {
        "grade_valid": False,
        "morphology_valid": False,
        "morphology_mapped": None,
        "tnm_valid": False,
        "laterality_valid": None,
        "treatment_valid": None,
        "comment": "Fallback due to non-JSON reviewer output; values require manual check."
    }

# ---------- prompt & chains ----------
RETURN_SPEC = """{
  "grade_valid": true or false,
  "morphology_valid": true or false,
  "morphology_mapped": "..." or null,
  "tnm_valid": true or false,
  "laterality_valid": true or false or null,
  "treatment_valid": true or false or null,
  "comment": "<summarize inconsistencies or mapping decisions>"
}"""

prompt = PromptTemplate(
    input_variables=["grade", "morphology", "tnm", "laterality", "treatment", "return_spec"],
    template="""
You are a clinical reviewer.

Validate and map the extracted fields from a pathology report. Output **ONLY** a single JSON object; no prose, no markdown, no code fences.

Input values:
- Tumor Grade: {grade}
- Morphology: {morphology}
- TNM Stage: {tnm}
- Laterality: {laterality}
- Treatment: {treatment}

Accepted Grade Values:
["Grade I", "Grade II", "Grade III", "Unknown"]

Accepted Morphology Values:
[
  "Adenocarcinoma, NOS",
  "Carcinoma, NOS",
  "Follicular adenocarcinoma, NOS",
  "Infiltrating duct carcinoma, NOS",
  "Lobular carcinoma, NOS",
  "Medullary carcinoma, NOS",
  "Mucinous adenocarcinoma",
  "Neoplasm, malignant",
  "Signet ring cell carcinoma",
  "Unknown"
]

Accepted TNM Components:
- T: ["T0", "Tis", "Tmic", "T1", "T2", "T3", "T4", "Tx", "Unknown"]
- N: ["N0", "N1", "N2", "N3", "Nx", "Unknown"]
- M: ["M0", "M1", "Mx", "Unknown"]

Accepted Laterality Values:
[
  "left, primary organ",
  "not a paired site/un",
  "paired, lat. unknown",
  "right, primary organ",
  "total colon",
  "unknown"
]

Accepted Treatment Values:
[
  "chemotherapy", "chemotherapy + radiotherapy", "chemotherapy + surgery",
  "chemotherapy + surgery + radiotherapy", "chemotherapy+hormonal",
  "chemotherapy+radiotherapy", "hormonal", "invalid code.",
  "neoadjuvant chemotherapy+surgery+radiotherapy+hormonal",
  "radical prostatectomy", "radiotherapy + surgery", "surgery",
  "surgery+chemotherapy", "surgery+chemotherapy+hormonal",
  "surgery+chemotherapy+radiotherapy",
  "surgery+chemotherapy+radiotherapy+hormonal", "surgery+hormonal",
  "surgery+radiotherapy+hormonal", "total thyroidectomy", "unknown"
]

Rules:
- If you are unsure, use null for *_valid fields that can be unknown, or false when it clearly doesn't match.
- DO NOT include any explanations; return JSON only.

Return JSON exactly in this shape:

{return_spec}
"""
)

# LLM -> plain string (we'll parse/repair ourselves)
review_chain = prompt | llm

# Repair prompt if LLM added prose or malformed JSON
repair_prompt = PromptTemplate.from_template("""
You will receive text that SHOULD be a JSON object with the following keys and types:

{return_spec}

Fix it if needed and return ONLY a valid JSON object. No comments, no markdown, no code fences.

Text:
{bad}
""")
repair_chain = repair_prompt | llm

def review_agent(state: Dict[str, Any]):
    grade = state.get("grade", "")
    morph = state.get("morphology", "")
    tnm = state.get("tnm", "")
    laterality = state.get("laterality", "")
    treatment = state.get("treatment", "")

    # 1) normal generate
    raw = (review_chain.invoke({
        "grade": grade,
        "morphology": morph,
        "tnm": tnm,
        "laterality": laterality,
        "treatment": treatment,
        "return_spec": RETURN_SPEC
    }) or "").strip()

    # 2) try to parse
    parsed = _extract_latest_json(raw)

    # 🔒 ensure dict (handle quoted-JSON or weird returns)
    if isinstance(parsed, str):
        try:
            maybe = json.loads(parsed)
            parsed = maybe if isinstance(maybe, dict) else {}
        except Exception:
            parsed = {}

    # 3) try to repair if needed
    if not isinstance(parsed, dict) or not parsed:
        fixed = (repair_chain.invoke({"return_spec": RETURN_SPEC, "bad": raw}) or "").strip()
        parsed = _extract_latest_json(fixed)
        if isinstance(parsed, str):
            try:
                maybe = json.loads(parsed)
                parsed = maybe if isinstance(maybe, dict) else {}
            except Exception:
                parsed = {}

    # 4) final fallback stub (never raise)
    if not isinstance(parsed, dict) or not parsed:
        parsed = _stub_review()
        log_message("review_agent", "warn", "Used stub review due to non-JSON output.")

    # 5) sanitize keys/types (works even if some keys missing)
    result = _sanitize_review(parsed)

    # Save to state (and keep raw/fixed for audit)
    ao = state.setdefault("agent_outputs", {})
    ao["review_agent_raw"] = raw
    ao["review_agent_fixed"] = parsed

    log_message("review_agent", "evaluator_agent", f"Validation results: {result}")
    state.update(result)
    return state
