from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mcp.middleware import log_message


# Initialize LLM
llm = OllamaLLM(model="llama3.3:latest")

# Prompt template for extracting tumor morphology
prompt = PromptTemplate.from_template(
    """
You are a medical language model specialized in extracting tumor morphology from pathology reports.
Your task is to extract and estimate the tumor **morphology** from the given report sections below.

Please output the following four key-value pairs:
1. "morphology stated": <The morphology as written in the pathology report, even if not standardized. If no morphology is directly stated, return "Unknown">,
2. "morphology estimated": <The closest standardized morphology based on the list below. Use your clinical knowledge to resolve synonymy or near matches. If unsure, return "Unknown">,
3. "morphology_CD": <Certainty degree of the estimation between 0.00 and 1.00>,
4. "morphology_evidence": <Key text or rationale supporting your estimation>

Valid standardized morphologies:
- Adenocarcinoma, NOS
- Carcinoma, NOS
- Follicular adenocarcinoma, NOS
- Infiltrating duct carcinoma, NOS
- Lobular carcinoma, NOS
- Medullary carcinoma, NOS
- Mucinous adenocarcinoma
- Neoplasm, malignant
- Signet ring cell carcinoma
- Unknown

Report Excerpt:
{context}

Output Format (JSON Only):
{{
  "morphology stated": "...",
  "morphology estimated": "...",
  "morphology_CD": ...,
  "morphology_evidence": "..."
}}
"""
)

# Construct the chain
chain = prompt | llm | StrOutputParser()


# Morphology extraction agent function
def morph_extractor(state):
    context = "\n".join(state["retrieved_chunks"].get("morphology", []))
    result = chain.invoke({"context": context}).strip()

    # Log raw output for traceability
    state.setdefault("agent_outputs", {})["morph_extractor"] = result

    # Update state with extracted morphology
    state["morphology"] = result

    # Optional logging for middleware/tracing
    log_message("morph_extractor", "review_agent", result)

    return state
