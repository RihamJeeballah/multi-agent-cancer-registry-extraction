from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mcp.middleware import log_message

llm = OllamaLLM(model="llama3.3:latest")

prompt = PromptTemplate.from_template(
    """
You are a clinical information extraction assistant. Your task is to extract the patient's **TNM stage** from the pathology report excerpt below.

Please extract the following key-value pairs for each TNM component: **T (Primary Tumor), N (Lymph Nodes), and M (Metastasis)**.

Each component should have the following 4 fields:

1. "<component> stated": <As written in the pathology report — e.g., "T2", "N1", "M0", or "Unknown">,
2. "<component> estimated": <Your best estimate based on AJCC 7th edition, if the stated value is missing or ambiguous. Use "Unknown" if not inferable>,
3. "<component>_CD": <Certainty degree of the estimation from 0.00 to 1.00>,
4. "<component>_evidence": <Supporting text or reasoning from the report>

Allowed values:

- **T (Primary Tumor)**: ["T0", "Tis", "Tmic", "T1", "T2", "T3", "T4", "Tx", "Unknown"]
- **N (Regional Lymph Nodes)**: ["N0", "N1", "N2", "N3", "Nx", "Unknown"]
- **M (Distant Metastasis)**: ["M0", "M1", "Mx", "Unknown"]

Report Sections:
{context}

Output Format (JSON Only):
{{
  "T stated": "...",
  "T estimated": "...",
  "T_CD": ...,
  "T_evidence": "...",
  "N stated": "...",
  "N estimated": "...",
  "N_CD": ...,
  "N_evidence": "...",
  "M stated": "...",
  "M estimated": "...",
  "M_CD": ...,
  "M_evidence": "..."
}}
"""
)

chain = prompt | llm | StrOutputParser()

def tnm_extractor(state):
    context = "\n".join(state["retrieved_chunks"].get("tnm", []))
    result = chain.invoke({"context": context}).strip()

    # ✅ Save raw LLM result for inspection
    state.setdefault("agent_outputs", {})["tnm_extractor"] = result

    # ✅ Store to main field for downstream use
    state["tnm"] = result

    log_message("tnm_extractor", "review_agent", result)
    return state
