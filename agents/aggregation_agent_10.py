# /home/riham/Desktop/llm_based_info_extraction/Agentic_medical_ie/agents/aggregation_agent_10.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from mcp.middleware import log_message
from typing import Dict, Any
import json, re

llm = OllamaLLM(model="llama3.3:latest", temperature=0.0)

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def _as_text(value: Any) -> str:
    if value is None: 
        return ""
    if isinstance(value, list):
        items = [_normalize_ws(str(x)) for x in value if _normalize_ws(str(x))]
        return " | ".join(items[:2])
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return _normalize_ws(str(value))
    return _normalize_ws(str(value))

def _first_str(value: Any) -> str:
    if value is None: 
        return ""
    if isinstance(value, list):
        for x in value:
            s = _normalize_ws(str(x))
            if s:
                return s
        return ""
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return _normalize_ws(str(value))
    return _normalize_ws(str(value))

def extract_latest_json(text_or_obj: Any) -> dict:
    """Extract a dict from raw text, quoted/fenced JSON, or dict/list inputs."""
    if isinstance(text_or_obj, dict):
        return text_or_obj
    if isinstance(text_or_obj, list):
        text = " ".join(_normalize_ws(str(x)) for x in text_or_obj if _normalize_ws(str(x)))
    else:
        text = "" if text_or_obj is None else str(text_or_obj)
    text = text.strip()
    if not text:
        return {}
    # 1) direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) fenced json blocks
    try:
        dict_candidates = re.findall(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, re.IGNORECASE)
    except Exception:
        dict_candidates = []
    # 3) any brace object
    try:
        dict_candidates += re.findall(r'({[\s\S]*?})', text)
    except Exception:
        pass
    for raw in reversed(dict_candidates):
        try:
            obj = json.loads(raw.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}

# ---------- JSON shape passed as a variable (avoid brace-escaping issues) ----------
json_template = {
    "grade": {"value": "...","stated": "...","estimated": "...","confidence": 0.0,"valid": True,"correct": True,"evidence": "...","comment": "..."},
    "morphology": {"value": "...","stated": "...","estimated": "...","confidence": 0.0,"valid": True,"correct": True,"evidence": "...","comment": "..."},
    "tnm": {"value": "...","confidence": 0.0,"valid": True,"correct": True,
            "T": {"stated": "...","estimated": "...","evidence": "..."},
            "N": {"stated": "...","estimated": "...","evidence": "..."},
            "M": {"stated": "...","estimated": "...","evidence": "..."},
            "comment": "..."},
    "laterality": {"value": "...","stated": "...","estimated": "...","confidence": 0.0,"valid": True,"correct": True,"evidence": "...","comment": "..."},
    "treatment": {"value": "...","stated": "...","estimated": "...","confidence": 0.0,"valid": True,"correct": True,"evidence": "...","comment": "..."},
    "summary_comment": "Final summary of confidence and interpretation."
}
JSON_SPEC = json.dumps(json_template, indent=2, ensure_ascii=False)

aggregation_prompt = PromptTemplate.from_template("""
You are a clinical summarization assistant.
Your task is to compile a structured report using STRICT JSON format.

⚠️ Important:
- DO NOT return markdown or explanation.
- DO NOT wrap the JSON in ```json or any formatting.
- Return ONLY valid JSON exactly matching this structure:

{json_spec}

Input variables:
Grade: {grade}, Stated: {grade_stated}, Estimated: {grade_estimated}, Confidence: {grade_confidence}, Valid: {grade_valid}, Correct: {grade_correct}, Evidence: {grade_evidence}
Morphology: {morphology}, Stated: {morphology_stated}, Estimated: {morphology_estimated}, Confidence: {morphology_confidence}, Valid: {morphology_valid}, Correct: {morphology_correct}, Evidence: {morphology_evidence}
TNM: {tnm}, Confidence: {tnm_confidence}, Valid: {tnm_valid}, Correct: {tnm_correct}
T: Stated: {t_stated}, Estimated: {t_estimated}, Evidence: {t_evidence}
N: Stated: {n_stated}, Estimated: {n_estimated}, Evidence: {n_evidence}
M: Stated: {m_stated}, Estimated: {m_estimated}, Evidence: {m_evidence}
Laterality: {laterality}, Stated: {laterality_stated}, Estimated: {laterality_estimated}, Confidence: {laterality_confidence}, Valid: {laterality_valid}, Correct: {laterality_correct}, Evidence: {laterality_evidence}
Treatment: {treatment}, Stated: {treatment_stated}, Estimated: {treatment_estimated}, Confidence: {treatment_confidence}, Valid: {treatment_valid}, Correct: {treatment_correct}, Evidence: {treatment_evidence}
""")
chain = aggregation_prompt | llm

# A tiny repair prompt to fix malformed JSON
repair_prompt = PromptTemplate.from_template("""
You will receive a model output that SHOULD be valid JSON matching this schema:

{json_spec}

The text may contain extra words, code fences, or invalid JSON (quotes/commas/booleans).
Return ONLY a corrected JSON object that strictly conforms to the schema above. No markdown, no explanation.

Text to fix:
{bad}
""")
repair_chain = repair_prompt | llm

def _make_stub(input_vars: Dict[str, Any]) -> Dict[str, Any]:
    """Last-resort valid JSON so we never drop a row."""
    def _b(name, default=False):
        v = input_vars.get(name, default)
        return bool(v)
    def _n(name, default=0):
        v = input_vars.get(name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    return {
        "grade": {
            "value": input_vars.get("grade", "") or "",
            "stated": input_vars.get("grade_stated", "") or "",
            "estimated": input_vars.get("grade_estimated", "") or "Unknown",
            "confidence": _n("grade_confidence", 0.0),
            "valid": _b("grade_valid", False),
            "correct": _b("grade_correct", False),
            "evidence": input_vars.get("grade_evidence", "") or "",
            "comment": ""
        },
        "morphology": {
            "value": input_vars.get("morphology", "") or "",
            "stated": input_vars.get("morphology_stated", "") or "",
            "estimated": input_vars.get("morphology_estimated", "") or "Unknown",
            "confidence": _n("morphology_confidence", 0.0),
            "valid": _b("morphology_valid", False),
            "correct": _b("morphology_correct", False),
            "evidence": input_vars.get("morphology_evidence", "") or "",
            "comment": ""
        },
        "tnm": {
            "value": input_vars.get("tnm", "") or "",
            "confidence": _n("tnm_confidence", 0.0),
            "valid": _b("tnm_valid", False),
            "correct": _b("tnm_correct", False),
            "T": {"stated": input_vars.get("t_stated", "") or "", "estimated": input_vars.get("t_estimated", "") or "", "evidence": input_vars.get("t_evidence", "") or ""},
            "N": {"stated": input_vars.get("n_stated", "") or "", "estimated": input_vars.get("n_estimated", "") or "", "evidence": input_vars.get("n_evidence", "") or ""},
            "M": {"stated": input_vars.get("m_stated", "") or "", "estimated": input_vars.get("m_estimated", "") or "", "evidence": input_vars.get("m_evidence", "") or ""},
            "comment": ""
        },
        "laterality": {
            "value": input_vars.get("laterality", "") or "",
            "stated": input_vars.get("laterality_stated", "") or "",
            "estimated": input_vars.get("laterality_estimated", "") or "not primary site/unpaired",
            "confidence": _n("laterality_confidence", 0.0),
            "valid": _b("laterality_valid", False),
            "correct": _b("laterality_correct", False),
            "evidence": input_vars.get("laterality_evidence", "") or "",
            "comment": ""
        },
        "treatment": {
            "value": input_vars.get("treatment", "") or "",
            "stated": input_vars.get("treatment_stated", "") or "",
            "estimated": input_vars.get("treatment_estimated", "") or "unknown",
            "confidence": _n("treatment_confidence", 0.0),
            "valid": _b("treatment_valid", False),
            "correct": _b("treatment_correct", False),
            "evidence": input_vars.get("treatment_evidence", "") or "",
            "comment": ""
        },
        "summary_comment": "Auto-generated fallback summary due to unparseable model output."
    }

def aggregation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    # Prepare inputs (robust to lists/strings/dicts)
    grade_obj = extract_latest_json(state.get("grade"))
    morph_obj = extract_latest_json(state.get("morphology"))
    tnm_obj  = extract_latest_json(state.get("tnm"))
    lat_obj  = extract_latest_json(state.get("laterality"))
    tre_obj  = extract_latest_json(state.get("treatment"))

    # 🔒 force dicts — in case extractors returned strings/odd values
    if not isinstance(grade_obj, dict): grade_obj = {}
    if not isinstance(morph_obj, dict): morph_obj = {}
    if not isinstance(tnm_obj, dict):  tnm_obj  = {}
    if not isinstance(lat_obj, dict):  lat_obj  = {}
    if not isinstance(tre_obj, dict):  tre_obj  = {}

    input_vars = {
        "grade": _as_text(state.get("grade")),
        "grade_stated": grade_obj.get("grade stated", "") or grade_obj.get("stated", "") or _first_str(state.get("grade")),
        "grade_estimated": grade_obj.get("grade estimated", "") or grade_obj.get("estimated", ""),
        "grade_confidence": state.get("grade_confidence", 0),
        "grade_valid": state.get("grade_valid", False),
        "grade_correct": state.get("grade_correct", False),
        "grade_evidence": grade_obj.get("grade_evidence", "") or grade_obj.get("evidence", ""),

        "morphology": _as_text(state.get("morphology")),
        "morphology_stated": morph_obj.get("morphology stated", "") or morph_obj.get("stated", "") or _first_str(state.get("morphology")),
        "morphology_estimated": morph_obj.get("morphology estimated", "") or morph_obj.get("estimated", ""),
        "morphology_confidence": state.get("morphology_confidence", 0),
        "morphology_valid": state.get("morphology_valid", False),
        "morphology_correct": state.get("morphology_correct", False),
        "morphology_evidence": morph_obj.get("morphology_evidence", "") or morph_obj.get("evidence", ""),

        "tnm": _as_text(state.get("tnm")),
        "tnm_confidence": state.get("tnm_confidence", 0),
        "tnm_valid": state.get("tnm_valid", False),
        "tnm_correct": state.get("tnm_correct", False),
        "t_stated": tnm_obj.get("T stated", "") or tnm_obj.get("T", {}).get("stated", ""),
        "t_estimated": tnm_obj.get("T estimated", "") or tnm_obj.get("T", {}).get("estimated", ""),
        "t_evidence": tnm_obj.get("T_evidence", "") or tnm_obj.get("T", {}).get("evidence", ""),
        "n_stated": tnm_obj.get("N stated", "") or tnm_obj.get("N", {}).get("stated", ""),
        "n_estimated": tnm_obj.get("N estimated", "") or tnm_obj.get("N", {}).get("estimated", ""),
        "n_evidence": tnm_obj.get("N_evidence", "") or tnm_obj.get("N", {}).get("evidence", ""),
        "m_stated": tnm_obj.get("M stated", "") or tnm_obj.get("M", {}).get("stated", ""),
        "m_estimated": tnm_obj.get("M estimated", "") or tnm_obj.get("M", {}).get("estimated", ""),
        "m_evidence": tnm_obj.get("M_evidence", "") or tnm_obj.get("M", {}).get("evidence", ""),

        "laterality": _as_text(state.get("laterality")),
        "laterality_stated": lat_obj.get("laterality stated", "") or lat_obj.get("stated", "") or _first_str(state.get("laterality")),
        "laterality_estimated": lat_obj.get("laterality estimated", "") or lat_obj.get("estimated", ""),
        "laterality_confidence": state.get("laterality_confidence", 0),
        "laterality_valid": state.get("laterality_valid", False),
        "laterality_correct": state.get("laterality_correct", False),
        "laterality_evidence": lat_obj.get("laterality_evidence", "") or lat_obj.get("evidence", ""),

        "treatment": _as_text(state.get("treatment")),
        "treatment_stated": tre_obj.get("treatment stated", "") or tre_obj.get("stated", "") or _first_str(state.get("treatment")),
        "treatment_estimated": tre_obj.get("treatment estimated", "") or tre_obj.get("estimated", ""),
        "treatment_confidence": state.get("treatment_confidence", 0),
        "treatment_valid": state.get("treatment_valid", False),
        "treatment_correct": state.get("treatment_correct", False),
        "treatment_evidence": tre_obj.get("treatment_evidence", "") or tre_obj.get("evidence", ""),
    }

    # 1) normal attempt
    raw_output = (chain.invoke({**input_vars, "json_spec": JSON_SPEC}) or "").strip()
    parsed_output = extract_latest_json(raw_output)

    # 2) repair attempt if needed
    if not parsed_output:
        repair_text = (repair_chain.invoke({"json_spec": JSON_SPEC, "bad": raw_output}) or "").strip()
        parsed_output = extract_latest_json(repair_text)

    # 3) last resort: stub (NEVER raise)
    if not parsed_output:
        parsed_output = _make_stub(input_vars)
        log_message("aggregation_agent", "warn", "Used stub structured summary due to unparseable output.")

    state["structured_summary"] = parsed_output
    state["summary"] = parsed_output.get("summary_comment", "")
    state.setdefault("agent_outputs", {})["aggregation_agent"] = parsed_output
    log_message("aggregation_agent", "result", "Structured summary generated.")
    return state
