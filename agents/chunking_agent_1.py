# chunking_agent.py
from typing import Dict, Any, List
from langchain_ollama import OllamaLLM
import json
import re
from memory.vector_store import create_vector_store

ALLOWED_FOCUS = {"grade", "morphology", "tnm", "laterality", "treatment", "other"}

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s

def _strip_invisibles(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\u200b", "").replace("\ufeff", "")

def extract_latest_dict(text: str) -> dict:
    if not text:
        return {}
    cleaned_full = _strip_invisibles(text)
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
        cleaned_full,
        flags=re.IGNORECASE,
    )
    for block in reversed(fenced_blocks):
        try:
            return json.loads(_strip_invisibles(block))
        except Exception:
            pass
    brace_candidates = re.findall(r"(\{[\s\S]*?\})", cleaned_full)
    for cand in reversed(brace_candidates):
        try:
            return json.loads(_strip_invisibles(cand))
        except Exception:
            pass
    try:
        return json.loads(_strip_code_fences(cleaned_full))
    except Exception:
        return {}

def normalize_chunk_key(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"chunk": []}
    if "chunk" in obj and isinstance(obj["chunk"], list):
        return obj
    for k in ("chunks", "Chunk", "CHUNK"):
        if k in obj and isinstance(obj[k], list):
            obj["chunk"] = obj[k]
            break
    obj.setdefault("chunk", [])
    return obj

def validate_and_prune(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    good: List[Dict[str, Any]] = []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        section = c.get("section")
        text = c.get("text")
        focus = c.get("focus")
        if isinstance(section, str) and isinstance(text, str) and isinstance(focus, str):
            f = focus.strip().lower()
            if f in ALLOWED_FOCUS:
                good.append({
                    "section": _normalize_ws(section)[:40],
                    "text": text.strip(),
                    "focus": f,
                })
    return good

def fallback_other_if_empty(raw_input_text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if chunks:
        return chunks
    excerpt = (raw_input_text or "").strip()
    excerpt = excerpt.splitlines()[0] if "\n" in excerpt else excerpt
    excerpt = excerpt[:300] if excerpt else "No clear clinical content."
    return [{
        "section": "other",
        "text": excerpt,
        "focus": "other",
    }]

def _build_prompt(report: str) -> str:
    json_template = {
        "chunk": [
            {
                "section": "string",
                "text": "string",
                "focus": "grade | morphology | tnm | laterality | treatment | other"
            }
        ]
    }
    return f"""You are a medical NLP system that segments ANY clinical text (even if it has no pathology-style headers) into semantically meaningful chunks.

## Output requirements
- Return ONLY a single JSON object (no markdown, no explanations).
- Schema (strict):
{{
  "chunk": [
    {{
      "section": "string",
      "text": "string",
      "focus": "grade | morphology | tnm | laterality | treatment | other"
    }}
  ]
}}
- The "chunk" array MUST be non-empty (never return an empty list).
- Each item MUST have all three fields.
- "focus" must be exactly one of: grade, morphology, tnm, laterality, treatment, other.

## Coverage & missing categories (very important)
- Never fabricate text.
- If a category (e.g., grade) is completely absent in the input, simply do NOT emit a chunk for that category. Downstream will treat missing categories as None.
- The overall output MUST still contain at least one chunk. If no focal info is found, emit a single fallback chunk:
  - section: "other"
  - text: a short verbatim excerpt from the input (e.g., the first 1–3 sentences or the most informative line)
  - focus: "other"

## General rules (works for messy text)
- The input may be free text, clinic letters, summaries, MDT notes, radiology-pathology blends, bullet lists, tables, or copy-paste artifacts.
- If there are no formal headers, infer a best-fit "section" from this shortlist:
  ["diagnosis","gross","microscopy","immuno","comment","staging","clinical","treatment","history","other"].
- If the text mixes topics, split into small focused chunks (~1–4 sentences) and assign the primary focus:
  • grade → tumor grading phrases (e.g., "grade 2", "moderately differentiated", "Gleason group 3")
  • morphology → histologic type/cell type (e.g., "invasive ductal carcinoma", "adenocarcinoma", "papillary carcinoma")
  • tnm → any T/N/M mentions, stage group, pathologic vs clinical, even partial like "pT3", "N1", "M0", or "indeterminate"
  • laterality → left/right/bilateral/unifocal/multifocal; single-site organs can be "not applicable" only if clearly stated
  • treatment → surgery, chemo, radio, hormonal, targeted, immuno; past or planned
  • other → demographics, dates, admin, unrelated text
- Keep each "text" verbatim (trim only leading/trailing whitespace and bullets). Do NOT invent placeholders like "none" or "[absent]".

## Normalization hints BEFORE deciding focus
- TNM appears as "pT3 N1 M0", "T3N1M0", "T3 N1 (Mx)", embedded in sentences, or on separate lines.
- Grades: prose ("moderately differentiated"), numerals ("grade II/2"), Gleason groups/scores.
- Laterality: "Lt", "Rt", "B/L", "bilat." map to left/right/bilateral.
- Morphology often near "diagnosis", "type", "histology", "consistent with".

## Edge cases & guarantees
- If you cannot find any obvious clinical focus, still emit the fallback "other" chunk (see Coverage).
- If TNM elements are scattered (e.g., T far from N), use multiple tnm chunks.
- Prefer short, precise chunks over one giant block.

Pathology/clinical text:
{report}

Here is the JSON template to follow exactly:
{json.dumps(json_template, indent=2)}
""".strip()

def chunking_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    report = (state.get("report") or "").strip()
    if not report:
        minimal = [{"section": "other", "text": "", "focus": "other"}]
        raw = json.dumps({"chunk": minimal})
        state.setdefault("agent_outputs", {})["chunker"] = raw
        state["agent_outputs"]["chunker_json"] = {"chunk": minimal}
        state["chunks"] = minimal
        state["vectorstore"] = create_vector_store([""], [{"section": "other", "focus": "other"}])
        return state

    prompt = _build_prompt(report)
    llm = OllamaLLM(model="llama3.3:latest", temperature=0.0, num_ctx=8192, keep_alive="5m")
    response = llm.invoke(prompt).strip()
    state.setdefault("agent_outputs", {})["chunker"] = response

    parsed = extract_latest_dict(response)
    parsed = normalize_chunk_key(parsed)
    cleaned = validate_and_prune(parsed.get("chunk", []))
    cleaned = fallback_other_if_empty(report, cleaned)

    state["agent_outputs"]["chunker_json"] = {"chunk": cleaned}
    chunk_texts = [c["text"] for c in cleaned]
    metadatas = [{"section": c["section"], "focus": c["focus"]} for c in cleaned]
    vectorstore = create_vector_store(chunk_texts, metadatas)

    state["chunks"] = cleaned
    state["vectorstore"] = vectorstore
    print("chunking_agent ran and added vectorstore with", len(cleaned), "chunks")
    return state
