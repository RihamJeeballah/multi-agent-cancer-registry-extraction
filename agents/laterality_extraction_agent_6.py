from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mcp.middleware import log_message

llm = OllamaLLM(model="llama3.3:latest")

prompt = PromptTemplate.from_template(
    """
You are an expert medical assistant specialized in cancer pathology information extraction.

Your task is to extract and estimate the **tumor laterality** from the provided pathology report excerpt below.

Please extract the following four key-value pairs related to the **tumor laterality**:

1. "laterality stated": <The laterality as explicitly mentioned in the pathology report: "left", "right", "bilateral involvement", or "not primary site/unpaired">,
2. "laterality estimated": <Your best estimate of laterality based on clinical knowledge. Use "not primary site/unpaired" if unclear or unpaired organ>,
3. "laterality_CD": <Certainty Degree between 0.00 and 1.00 reflecting confidence in the estimation>,
4. "laterality_evidence": <A short explanation or quote from the input text that supports your estimation>

Valid laterality values: ["left", "right", "bilateral involvement", "not primary site/unpaired"]

Input:
{context}

Output Format (JSON Only):
{{
  "laterality stated": "...",
  "laterality estimated": "...",
  "laterality_CD": ...,
  "laterality_evidence": "..."
}}
"""
)

chain = prompt | llm | StrOutputParser()

def laterality_extractor(state):
    context = "\n".join(state["retrieved_chunks"].get("laterality", []))
    result = chain.invoke({"context": context}).strip()

    state.setdefault("agent_outputs", {})["laterality_extractor"] = result
    state["laterality"] = result

    log_message("laterality_extractor", "review_agent", result)
    return state
