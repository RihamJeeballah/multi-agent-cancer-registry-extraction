from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from mcp.middleware import log_message

llm = OllamaLLM(model="llama3.3:latest", temperature=0.0)

# -----------------------------
# Confidence Prompt Template
# -----------------------------
confidence_prompt = PromptTemplate.from_template("""
You are a clinical NLP evaluator.

Your task is to estimate the confidence score (from 0 to 100) for the accuracy of each extracted field from a pathology report:
- Tumor **grade**
- Tumor **morphology**
- **TNM stage**
- **Laterality**
- **Treatment**

Base your estimate on:
- Whether the extracted value is complete and clinically valid
- Whether it follows accepted terminology
- Whether it is likely justified based on context

Input Fields:
- Grade: {grade}
- Morphology: {morphology}
- TNM: {tnm}
- Laterality: {laterality}
- Treatment: {treatment}

Output Format (JSON Only):
{{
  "grade_confidence": integer (0-100),
  "morphology_confidence": integer (0-100),
  "tnm_confidence": integer (0-100),
  "laterality_confidence": integer (0-100),
  "treatment_confidence": integer (0-100)
}}
""")

llm_chain = confidence_prompt | llm | JsonOutputParser()

# -----------------------------
# Evaluator Agent
# -----------------------------
def evaluator_agent(state):
    # Extracted values (normalize to lowercase for comparison)
    extracted_grade = state.get("grade", "").strip().lower()
    extracted_morphology = state.get("morphology", "").strip().lower()
    extracted_tnm = state.get("tnm", "").strip().lower()
    extracted_laterality = state.get("laterality", "").strip().lower()
    extracted_treatment = state.get("treatment", "").strip().lower()

    # Ground truth values from Streamlit input
    true_grade = state.get("true_grade")
    true_morphology = state.get("true_morphology")
    true_tnm = state.get("true_tnm")
    true_laterality = state.get("LATER")
    true_treatment = state.get("TRE-1")

    # Step 1: Compare with ground truth
    if true_grade:
        state["grade_correct"] = extracted_grade == true_grade.strip().lower()
        log_message("evaluator_agent", "result", f"Grade correct: {state['grade_correct']}")
    if true_morphology:
        state["morphology_correct"] = extracted_morphology == true_morphology.strip().lower()
        log_message("evaluator_agent", "result", f"Morphology correct: {state['morphology_correct']}")
    if true_tnm:
        state["tnm_correct"] = extracted_tnm == true_tnm.strip().lower()
        log_message("evaluator_agent", "result", f"TNM correct: {state['tnm_correct']}")
    if true_laterality:
        state["laterality_correct"] = extracted_laterality == true_laterality.strip().lower()
        log_message("evaluator_agent", "result", f"Laterality correct: {state['laterality_correct']}")
    if true_treatment:
        state["treatment_correct"] = extracted_treatment == true_treatment.strip().lower()
        log_message("evaluator_agent", "result", f"Treatment correct: {state['treatment_correct']}")

    # Step 2: LLM Confidence (only if any ground truth is missing)
    if not all([true_grade, true_morphology, true_tnm, true_laterality, true_treatment]):
        try:
            confidence = llm_chain.invoke({
                "grade": extracted_grade,
                "morphology": extracted_morphology,
                "tnm": extracted_tnm,
                "laterality": extracted_laterality,
                "treatment": extracted_treatment
            })
        except Exception as e:
            log_message("evaluator_agent", "error", f"Confidence estimation failed: {e}")
            confidence = {
                "grade_confidence": 0,
                "morphology_confidence": 0,
                "tnm_confidence": 0,
                "laterality_confidence": 0,
                "treatment_confidence": 0
            }

        state.update(confidence)
        state.setdefault("agent_outputs", {})["evaluator_agent"] = confidence
        log_message("evaluator_agent", "confidence_scores", confidence)

    return state
