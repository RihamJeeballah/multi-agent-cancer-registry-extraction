import streamlit as st
import pandas as pd
from memory.long_term_memory import save_extraction_to_memory, Session, Extraction
from workflows.document_workflow import build_workflow
from mcp.protocol import protocol_instance
from langgraph.graph import START, END
from pyvis.network import Network
import tempfile
from agents.aggregation_agent_10 import aggregation_agent

compiled_graph, raw_graph = build_workflow()

# -------------------------
# DB + Workflow Functions
# -------------------------

def run_workflow_on_report(report_text):
    protocol_instance.messages.clear()
    initial_state = {
        "report": report_text,
        "question": "Extract the tumor grade, morphology, TNM stage, laterality, and treatment from this patient's pathology report.",
        "chunks": [],
        "retrieved_chunks": {},
        "grade": "",
        "morphology": "",
        "tnm": "",
        "laterality": "",
        "treatment": "",
        "grade_valid": False,
        "morphology_valid": False,
        "tnm_valid": False,
        "laterality_valid": False,
        "treatment_valid": False,
        "grade_confidence": 0.0,
        "morphology_confidence": 0.0,
        "tnm_confidence": 0.0,
        "laterality_confidence": 0.0,
        "treatment_confidence": 0.0,
        "grade_correct": False,
        "morphology_correct": False,
        "tnm_correct": False,
        "laterality_correct": False,
        "treatment_correct": False,
        "summary": "",
        "mcp_log": []
    }

    result = compiled_graph.invoke(initial_state)
    result["mcp_log"] = [
        {
            "from": msg.sender,
            "to": msg.receiver,
            "content": msg.content,
            "confidence": msg.confidence,
        }
        for msg in protocol_instance.get_history()
    ]
    return result


def visualize_workflow():
    net = Network(directed=True, height="700px", width="100%", bgcolor="#222222", font_color="white")
    for node in raw_graph.nodes:
        net.add_node(node, label=node)
    net.add_node(START, label="START", color="green", shape="box")
    net.add_node(END, label="END", color="red", shape="box")
    for src, tgt in raw_graph.edges:
        if not isinstance(tgt, (list, tuple)):
            tgt = [tgt]
        for t in tgt:
            net.add_edge(src, t)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp.name)
    return tmp.name

# -------------------------
# Streamlit UI
# -------------------------

st.title("Agentic Cancer Report Extractor")
st.markdown("Upload a pathology report or CSV to extract tumor grade, morphology, TNM stage, laterality, and treatment.")

option = st.radio("Choose input method:", ["Single Report", "CSV File"])

if option == "Single Report":
    user_input = st.text_area("Paste pathology report text here:")
    patient_id = st.text_input("Enter patient MRN (Medical Record Number):")
    if st.button("Run Analysis") and user_input.strip() and patient_id.strip():
        with st.spinner("Running agentic pipeline..."):
            state = run_workflow_on_report(user_input)
            state["true_grade"] = None
            state["true_morphology"] = None
            state["true_tnm"] = None
            state["LATER"] = None
            state["TRE-1"] = None
            state = aggregation_agent(state)

            save_extraction_to_memory(
                state=state,
                mrn=patient_id,
                agent_version="v1.0",
                grade_prompt_version="grade_prompt_v1",
                morph_prompt_version="morph_prompt_v1",
                tnm_prompt_version="tnm_prompt_v1"
            )
        st.success("✅ Extraction complete and saved.")
        st.json(state.get("structured_summary"))

elif option == "CSV File":
    uploaded_file = st.file_uploader("Upload CSV file with a 'report' and 'mrn' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "report" not in df.columns or "mrn" not in df.columns:
            st.error("CSV must contain both 'report' and 'mrn' columns.")
        else:
            st.info(f"Detected {len(df)} reports.")
            if st.button("Process All"):
                with st.spinner("Processing all reports..."):
                    skipped = []
                    for i, row in df.iterrows():
                        mrn = str(row["mrn"])
                        report = row["report"]
                        st.write(f"🔄 Processing row {i+1} — MRN: {mrn}")

                        try:
                            state = run_workflow_on_report(report)

                            # Inject ground truth
                            state["true_grade"] = str(row["GRADE (desc)"]) if "GRADE (desc)" in row and pd.notna(row["GRADE (desc)"]) else None
                            state["true_morphology"] = str(row["MORP (desc)"]) if "MORP (desc)" in row and pd.notna(row["MORP (desc)"]) else None
                            if all(k in row and pd.notna(row[k]) for k in ["T_cleaned", "N_cleaned", "M_cleaned"]):
                                state["true_tnm"] = f"{row['T_cleaned']}{row['N_cleaned']}{row['M_cleaned']}"
                                state["T_cleaned"] = row["T_cleaned"]
                                state["N_cleaned"] = row["N_cleaned"]
                                state["M_cleaned"] = row["M_cleaned"]
                            state["true_laterality"] = str(row["LATER"]).strip() if "LATER" in row and pd.notna(row["LATER"]) and str(row["LATER"]).strip() else None
                            state["true_treatment"] = str(row["TRE-1"]).strip() if "TRE-1" in row and pd.notna(row["TRE-1"]) and str(row["TRE-1"]).strip() else None

                            # Aggregation
                            state = aggregation_agent(state)

                            # Save to DB
                            save_extraction_to_memory(
                                state=state,
                                mrn=mrn,
                                agent_version="v1.0",
                                grade_prompt_version="grade_prompt_v1",
                                morph_prompt_version="morph_prompt_v1",
                                tnm_prompt_version="tnm_prompt_v1",
                                laterality_prompt_version="laterality_prompt_v1",
                                treatment_prompt_version="treatment_prompt_v1"
                            )

                        except Exception as e:
                            skipped.append(mrn)
                            st.warning(f"⚠️ Skipping MRN {mrn} (row {i+1}) due to error: {e}")
                            continue

                    st.success("✅ All extractions completed.")
                    if skipped:
                        st.warning(f"⚠️ Skipped {len(skipped)} record(s): {', '.join(skipped[:10])}...")


if st.checkbox("Show workflow graph"):
    html_path = visualize_workflow()
    st.components.v1.html(open(html_path, 'r', encoding='utf-8').read(), height=800)

if st.checkbox("View Stored Extractions"):
    session = Session()
    all_extractions = session.query(Extraction).order_by(Extraction.timestamp.desc()).all()
    session.close()

    if not all_extractions:
        st.warning("No extractions found in the database.")
    else:
        for ext in all_extractions:
            with st.expander(f"Patient ID: {ext.patient_id} — {ext.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                st.markdown("### Tumor Grade")
                st.write({
                    "Grade": ext.grade,
                    "Stated": ext.grade_stated,
                    "Estimated": ext.grade_estimated,
                    "Evidence": ext.grade_evidence,
                    "Confidence": ext.grade_confidence,
                    "Valid": ext.grade_valid,
                    "Correct": ext.grade_correct,
                    "Ground Truth": ext.grade_gold,
                    "Feedback": ext.grade_feedback,
                    "Comment": ext.grade_comment,
                    "Prompt Version": ext.grade_prompt_version,
                })

                st.markdown("### Morphology")
                st.write({
                    "Morphology": ext.morphology,
                    "Stated": ext.morphology_stated,
                    "Estimated": ext.morphology_estimated,
                    "Evidence": ext.morphology_evidence,
                    "Confidence": ext.morph_confidence,
                    "Valid": ext.morphology_valid,
                    "Correct": ext.morph_correct,
                    "Ground Truth": ext.morph_gold,
                    "Feedback": ext.morph_feedback,
                    "Comment": ext.morphology_comment,
                    "Prompt Version": ext.morph_prompt_version,
                })

                st.markdown("### TNM Stage")
                st.write({
                    "TNM": ext.tnm,
                    "T": {
                        "Stated": ext.t_stated,
                        "Estimated": ext.t_estimated,
                        "Evidence": ext.t_evidence,
                    },
                    "N": {
                        "Stated": ext.n_stated,
                        "Estimated": ext.n_estimated,
                        "Evidence": ext.n_evidence,
                    },
                    "M": {
                        "Stated": ext.m_stated,
                        "Estimated": ext.m_estimated,
                        "Evidence": ext.m_evidence,
                    },
                    "Confidence": ext.tnm_confidence,
                    "Valid": ext.tnm_valid,
                    "Correct": ext.tnm_correct,
                    "Ground Truth": ext.tnm_gold,
                    "Feedback": ext.tnm_feedback,
                    "Comment": ext.tnm_comment,
                    "Prompt Version": ext.tnm_prompt_version,
                })

                st.markdown("### Laterality")
                st.write({
                    "Laterality": ext.laterality,
                    "Stated": ext.laterality_stated,
                    "Estimated": ext.laterality_estimated,
                    "Evidence": ext.laterality_evidence,
                    "Confidence": ext.laterality_confidence,
                    "Valid": ext.laterality_valid,
                    "Correct": ext.laterality_correct,
                    "Ground Truth": ext.laterality_gold,
                    "Feedback": ext.laterality_feedback,
                    "Comment": ext.laterality_comment,
                })

                st.markdown("### Treatment")
                st.write({
                    "Treatment": ext.treatment,
                    "Stated": ext.treatment_stated,
                    "Estimated": ext.treatment_estimated,
                    "Evidence": ext.treatment_evidence,
                    "Confidence": ext.treatment_confidence,
                    "Valid": ext.treatment_valid,
                    "Correct": ext.treatment_correct,
                    "Ground Truth": ext.treatment_gold,
                    "Feedback": ext.treatment_feedback,
                    "Comment": ext.treatment_comment,
                })

                st.markdown("### Final Summary")
                st.write({
                    "Summary": ext.summary,
                    "Comment": ext.summary_comment,
                    "Tags": ext.tags,
                })

# Export
if st.checkbox("Export Stored Results to CSV"):
    session = Session()
    all_extractions = session.query(Extraction).all()
    session.close()

    if not all_extractions:
        st.warning("No extractions found to export.")
    else:
        data = [
            {
                "mrn": ext.patient_id,
                "timestamp": ext.timestamp,
                "report": ext.report,
                "grade": ext.grade,
                "grade_stated": ext.grade_stated,
                "grade_estimated": ext.grade_estimated,
                "grade_confidence": ext.grade_confidence,
                "grade_correct": ext.grade_correct,
                "grade_gold": ext.grade_gold,
                "morphology": ext.morphology,
                "morphology_stated": ext.morphology_stated,
                "morphology_estimated": ext.morphology_estimated,
                "morph_confidence": ext.morph_confidence,
                "morph_correct": ext.morph_correct,
                "morph_gold": ext.morph_gold,
                "tnm": ext.tnm,
                "tnm_confidence": ext.tnm_confidence,
                "tnm_correct": ext.tnm_correct,
                "tnm_gold": ext.tnm_gold,
                "laterality": ext.laterality,
                "laterality_stated": ext.laterality_stated,
                "laterality_estimated": ext.laterality_estimated,
                "laterality_confidence": ext.laterality_confidence,
                "laterality_correct": ext.laterality_correct,
                "laterality_gold": ext.laterality_gold,
                "treatment": ext.treatment,
                "treatment_stated": ext.treatment_stated,
                "treatment_estimated": ext.treatment_estimated,
                "treatment_confidence": ext.treatment_confidence,
                "treatment_correct": ext.treatment_correct,
                "treatment_gold": ext.treatment_gold,
                "summary": ext.summary,
                "summary_comment": ext.summary_comment,
            }
            for ext in all_extractions
        ]
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="extracted_results.csv",
            mime="text/csv",
        )