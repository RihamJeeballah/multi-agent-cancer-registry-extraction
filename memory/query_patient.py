import sys
from pathlib import Path

# Add project root to system path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Agentic_medical_ie.memory.long_term_memory import get_extractions_by_patient

def display_extractions(mrn: str):
    extractions = get_extractions_by_patient(mrn)

    if not extractions:
        print(f"No extractions found for MRN: {mrn}")
        return

    for ext in extractions:
        print("\nExtraction Summary")
        print(f"MRN                : {ext.patient_id}")
        print(f"Timestamp          : {ext.timestamp}")
        print(f"Agent Version      : {ext.agent_version}")

        print("\n--- Tumor Grade ---")
        print(f"Grade              : {ext.grade}")
        print(f"Stated             : {ext.grade_stated}")
        print(f"Estimated          : {ext.grade_estimated}")
        print(f"Evidence           : {ext.grade_evidence}")
        print(f"Confidence         : {ext.grade_confidence}")
        print(f"Valid              : {ext.grade_valid}")
        print(f"Correct            : {ext.grade_correct}")
        print(f"Ground Truth       : {ext.grade_gold}")
        print(f"Feedback           : {ext.grade_feedback}")
        print(f"Comment            : {ext.grade_comment}")
        print(f"Prompt Version     : {ext.grade_prompt_version}")

        print("\n--- Morphology ---")
        print(f"Morphology         : {ext.morphology}")
        print(f"Stated             : {ext.morphology_stated}")
        print(f"Estimated          : {ext.morphology_estimated}")
        print(f"Evidence           : {ext.morphology_evidence}")
        print(f"Confidence         : {ext.morph_confidence}")
        print(f"Valid              : {ext.morphology_valid}")
        print(f"Correct            : {ext.morph_correct}")
        print(f"Ground Truth       : {ext.morph_gold}")
        print(f"Feedback           : {ext.morph_feedback}")
        print(f"Comment            : {ext.morphology_comment}")
        print(f"Prompt Version     : {ext.morph_prompt_version}")

        print("\n--- TNM Stage ---")
        print(f"TNM                : {ext.tnm}")
        print(f"T: Stated          : {ext.t_stated}, Estimated: {ext.t_estimated}, Evidence: {ext.t_evidence}")
        print(f"N: Stated          : {ext.n_stated}, Estimated: {ext.n_estimated}, Evidence: {ext.n_evidence}")
        print(f"M: Stated          : {ext.m_stated}, Estimated: {ext.m_estimated}, Evidence: {ext.m_evidence}")
        print(f"Confidence         : {ext.tnm_confidence}")
        print(f"Valid              : {ext.tnm_valid}")
        print(f"Correct            : {ext.tnm_correct}")
        print(f"Ground Truth       : {ext.tnm_gold}")
        print(f"Feedback           : {ext.tnm_feedback}")
        print(f"Comment            : {ext.tnm_comment}")
        print(f"Prompt Version     : {ext.tnm_prompt_version}")

        print("\n--- Summary ---")
        print(f"Summary            : {ext.summary}")
        print(f"Comment            : {ext.summary_comment}")
        print(f"Tags               : {ext.tags}")
        print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mrn = sys.argv[1]
    else:
        mrn = "P001"
    display_extractions(mrn)
