from typing import Dict, Any, List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json
import re

# --- helpers: robust array extraction (accepts empty []) ---
def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s

def _extract_json_array(text: str) -> List[str]:
    """
    Return a JSON array of strings from an LLM response.
    - Accept [] as valid.
    - Prefer fully-parsed whole-string JSON.
    - Otherwise, try last fenced block, then last []-block.
    - If parsing fails or result is not a list, return [] instead of raising.
    """
    if not isinstance(text, str):
        return []

    cleaned = text.strip()
    # 1) exact parse (whole-string)
    try:
        obj = json.loads(_strip_fences(cleaned))
        if isinstance(obj, list):
            # keep only strings; ignore non-strings quietly
            return [str(x) for x in obj if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception:
        pass

    # 2) fenced blocks
    blocks = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", cleaned, flags=re.IGNORECASE)
    for raw in reversed(blocks):
        try:
            arr = json.loads(raw.strip())
            if isinstance(arr, list):
                return [str(x) for x in arr if isinstance(x, (str, int, float)) and str(x).strip()]
        except Exception:
            continue

    # 3) any bracketed array (including empty [])
    arrays = re.findall(r"(\[[\s\S]*?\])", cleaned)
    for raw in reversed(arrays):
        try:
            arr = json.loads(raw.strip())
            if isinstance(arr, list):
                return [str(x) for x in arr if isinstance(x, (str, int, float)) and str(x).strip()]
        except Exception:
            continue

    # nothing usable
    return []

def retriever_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"retriever_agent received state keys: {list(state.keys())}")

    chunks = state.get("chunks")
    if not chunks:
        raise ValueError("Chunks not found in state. Make sure the chunking agent ran first.")

    llm = OllamaLLM(model="llama3.3:latest")

    retrieved_chunks: Dict[str, List[str]] = {}

    for field in ["grade", "morphology", "tnm", "laterality", "treatment"]:
        prompt_template = PromptTemplate.from_template("""
You are a medical AI assistant helping extract information from pathology report chunks.

Goal: Find the chunks most relevant to the field: **{field}**.

Chunks:
{chunks_text}

Instructions:
- Return only the most relevant chunks for the field: "{field}"
- Return ONLY a JSON array of strings (chunk texts). If none are relevant, return [].
- No explanation, no markdown, no preamble

Example outputs:
[
  "Grade 2 tumor with moderate atypia.",
  "Another relevant chunk text."
]
or
[]
""")
        formatted_chunks = "\n".join(
            [f"- Section: {c['section']} | Focus: {c['focus']} | Text: {c['text']}" for c in chunks]
        )
        prompt = prompt_template.format(field=field, chunks_text=formatted_chunks)

        response = llm.invoke(prompt).strip()
        print(f"\n🔍 Raw response for field '{field}':\n{response}\n")

        # ✅ tolerant extraction: accepts [] and mixed formatting
        arr = _extract_json_array(response)
        # Don’t raise if empty — empty is valid (means nothing relevant)
        retrieved_chunks[field] = arr  # may be []

    state["retrieved_chunks"] = retrieved_chunks
    return state
