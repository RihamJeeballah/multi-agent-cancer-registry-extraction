from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mcp.middleware import log_message

llm = OllamaLLM(model="llama3.3:latest")

prompt = PromptTemplate.from_template(
    """
You are a clinical assistant trained to extract information about treatments received by cancer patients.

Based on the pathology report excerpt provided below, please answer whether the patient underwent the following treatments.

Answer each of the following questions with "Yes", "No", or "Unclear". Also provide a certainty score and justification.

Treatments:
1. Surgery
2. Chemotherapy
3. Radiotherapy
4. Hormonal therapy
5. Immunotherapy

Input:
{context}

Output Format (JSON Only):
{{
  "Surgery": {{
    "answer": "...",
    "certainty": ...,
    "evidence": "..."
  }},
  "Chemotherapy": {{
    "answer": "...",
    "certainty": ...,
    "evidence": "..."
  }},
  "Radiotherapy": {{
    "answer": "...",
    "certainty": ...,
    "evidence": "..."
  }},
  "Hormonal therapy": {{
    "answer": "...",
    "certainty": ...,
    "evidence": "..."
  }},
  "Immunotherapy": {{
    "answer": "...",
    "certainty": ...,
    "evidence": "..."
  }}
}}
"""
)

chain = prompt | llm | StrOutputParser()

def treatment_extractor(state):
    context = "\n".join(state["retrieved_chunks"].get("treatment", []))
    result = chain.invoke({"context": context}).strip()

    state.setdefault("agent_outputs", {})["treatment_extractor"] = result
    state["treatment"] = result

    log_message("treatment_extractor", "review_agent", result)
    return state
