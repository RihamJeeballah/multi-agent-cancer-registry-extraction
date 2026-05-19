from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
DB_PATH = MEMORY_DIR / "agentic_memory.db"

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# ------------------------
# Table 1: Prompt Registry
# ------------------------

class Prompt(Base):
    __tablename__ = "prompts"

    version_id = Column(String, primary_key=True)
    field = Column(String)
    template = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prompt(version={self.version_id}, field={self.field})>"

# ------------------------
# Table 2: Extractions
# ------------------------

class Extraction(Base):
    __tablename__ = "extractions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    report = Column(Text)

    # Grade
    grade = Column(String)
    grade_stated = Column(String)
    grade_estimated = Column(String)
    grade_evidence = Column(Text)
    grade_confidence = Column(Float)
    grade_valid = Column(String)
    grade_correct = Column(String)
    grade_gold = Column(String)
    grade_feedback = Column(String)
    grade_comment = Column(Text)

    # Morphology
    morphology = Column(String)
    morphology_stated = Column(String)
    morphology_estimated = Column(String)
    morphology_evidence = Column(Text)
    morph_confidence = Column(Float)
    morphology_valid = Column(String)
    morph_correct = Column(String)
    morph_gold = Column(String)
    morph_feedback = Column(String)
    morphology_comment = Column(Text)

    # TNM
    tnm = Column(String)
    t_stated = Column(String)
    t_estimated = Column(String)
    t_evidence = Column(Text)
    n_stated = Column(String)
    n_estimated = Column(String)
    n_evidence = Column(Text)
    m_stated = Column(String)
    m_estimated = Column(String)
    m_evidence = Column(Text)
    tnm_confidence = Column(Float)
    tnm_valid = Column(String)
    tnm_correct = Column(String)
    tnm_gold = Column(String)
    tnm_feedback = Column(String)
    tnm_comment = Column(Text)

    # Laterality
    laterality = Column(String)
    laterality_stated = Column(String)
    laterality_estimated = Column(String)
    laterality_evidence = Column(Text)
    laterality_confidence = Column(Float)
    laterality_valid = Column(String)
    laterality_correct = Column(String)
    laterality_gold = Column(String)
    laterality_feedback = Column(String)
    laterality_comment = Column(Text)

    # Treatment
    treatment = Column(String)
    treatment_stated = Column(String)
    treatment_estimated = Column(String)
    treatment_evidence = Column(Text)
    treatment_confidence = Column(Float)
    treatment_valid = Column(String)
    treatment_correct = Column(String)
    treatment_gold = Column(String)
    treatment_feedback = Column(String)
    treatment_comment = Column(Text)

    # Summary and Tags
    summary = Column(Text)
    summary_comment = Column(Text)
    tags = Column(String)

    # Versioning
    agent_version = Column(String)
    grade_prompt_version = Column(String, ForeignKey("prompts.version_id"))
    morph_prompt_version = Column(String, ForeignKey("prompts.version_id"))
    tnm_prompt_version = Column(String, ForeignKey("prompts.version_id"))
    laterality_prompt_version = Column(String, ForeignKey("prompts.version_id"))
    treatment_prompt_version = Column(String, ForeignKey("prompts.version_id"))

    grade_prompt = relationship("Prompt", foreign_keys=[grade_prompt_version])
    morph_prompt = relationship("Prompt", foreign_keys=[morph_prompt_version])
    tnm_prompt = relationship("Prompt", foreign_keys=[tnm_prompt_version])
    laterality_prompt = relationship("Prompt", foreign_keys=[laterality_prompt_version])
    treatment_prompt = relationship("Prompt", foreign_keys=[treatment_prompt_version])

    def __repr__(self):
        return f"<Extraction(patient={self.patient_id}, grade={self.grade})>"

# ------------------------
# Utility Functions
# ------------------------

def init_db():
    Base.metadata.create_all(engine)

def get_extractions_by_patient(patient_id: str):
    session = Session()
    extractions = session.query(Extraction).filter(Extraction.patient_id == patient_id).all()
    session.close()
    return extractions

def save_extraction_to_memory(
    state: dict,
    mrn: str,
    agent_version: str,
    grade_prompt_version: str,
    morph_prompt_version: str,
    tnm_prompt_version: str,
    laterality_prompt_version: str,
    treatment_prompt_version: str
):
    session = Session()
    summary = state.get("structured_summary", {})
    if not summary:
        print(f"⚠️ No structured summary for MRN {mrn}, skipping.")
        session.close()
        return

    session.query(Extraction).filter(Extraction.patient_id == mrn).delete()

    treatment_data = summary.get("treatment", {})
    treatment_value = treatment_data.get("value")
    if isinstance(treatment_value, dict):
        treatment_value = str(treatment_value)

    extraction = Extraction(
        patient_id=mrn,
        agent_version=agent_version,
        grade_prompt_version=grade_prompt_version,
        morph_prompt_version=morph_prompt_version,
        tnm_prompt_version=tnm_prompt_version,
        laterality_prompt_version=laterality_prompt_version,
        treatment_prompt_version=treatment_prompt_version,

        report=state.get("report"),

        # Grade
        grade=summary.get("grade", {}).get("value"),
        grade_stated=summary.get("grade", {}).get("stated"),
        grade_estimated=summary.get("grade", {}).get("estimated"),
        grade_evidence=summary.get("grade", {}).get("evidence"),
        grade_confidence=summary.get("grade", {}).get("confidence"),
        grade_valid=str(summary.get("grade", {}).get("valid")),
        grade_correct=str(summary.get("grade", {}).get("correct")),
        grade_feedback=state.get("grade_feedback"),
        grade_gold=state.get("true_grade"),
        grade_comment=summary.get("grade", {}).get("comment"),

        # Morphology
        morphology=summary.get("morphology", {}).get("value"),
        morphology_stated=summary.get("morphology", {}).get("stated"),
        morphology_estimated=summary.get("morphology", {}).get("estimated"),
        morphology_evidence=summary.get("morphology", {}).get("evidence"),
        morph_confidence=summary.get("morphology", {}).get("confidence"),
        morphology_valid=str(summary.get("morphology", {}).get("valid")),
        morph_correct=str(summary.get("morphology", {}).get("correct")),
        morph_feedback=state.get("morphology_feedback"),
        morph_gold=state.get("true_morphology"),
        morphology_comment=summary.get("morphology", {}).get("comment"),

        # TNM
        tnm=summary.get("tnm", {}).get("value"),
        t_stated=summary.get("tnm", {}).get("T", {}).get("stated"),
        t_estimated=summary.get("tnm", {}).get("T", {}).get("estimated"),
        t_evidence=summary.get("tnm", {}).get("T", {}).get("evidence"),
        n_stated=summary.get("tnm", {}).get("N", {}).get("stated"),
        n_estimated=summary.get("tnm", {}).get("N", {}).get("estimated"),
        n_evidence=summary.get("tnm", {}).get("N", {}).get("evidence"),
        m_stated=summary.get("tnm", {}).get("M", {}).get("stated"),
        m_estimated=summary.get("tnm", {}).get("M", {}).get("estimated"),
        m_evidence=summary.get("tnm", {}).get("M", {}).get("evidence"),
        tnm_confidence=summary.get("tnm", {}).get("confidence"),
        tnm_valid=str(summary.get("tnm", {}).get("valid")),
        tnm_correct=str(summary.get("tnm", {}).get("correct")),
        tnm_feedback=state.get("tnm_feedback"),
        tnm_gold=state.get("true_tnm"),
        tnm_comment=summary.get("tnm", {}).get("comment"),

        # Laterality
        laterality=summary.get("laterality", {}).get("value"),
        laterality_stated=summary.get("laterality", {}).get("stated"),
        laterality_estimated=summary.get("laterality", {}).get("estimated"),
        laterality_evidence=summary.get("laterality", {}).get("evidence"),
        laterality_confidence=summary.get("laterality", {}).get("confidence"),
        laterality_valid=str(summary.get("laterality", {}).get("valid")),
        laterality_correct=str(summary.get("laterality", {}).get("correct")),
        laterality_feedback=state.get("laterality_feedback"),
        laterality_gold=state.get("true_laterality"),
        laterality_comment=summary.get("laterality", {}).get("comment"),

        # Treatment
        treatment=treatment_value,
        treatment_stated=treatment_data.get("stated"),
        treatment_estimated=treatment_data.get("estimated"),
        treatment_evidence=treatment_data.get("evidence"),
        treatment_confidence=treatment_data.get("confidence"),
        treatment_valid=str(treatment_data.get("valid")),
        treatment_correct=str(treatment_data.get("correct")),
        treatment_feedback=state.get("treatment_feedback"),
        treatment_gold=state.get("true_treatment"),
        treatment_comment=treatment_data.get("comment"),

        # Summary
        summary=state.get("summary"),
        summary_comment=summary.get("summary_comment"),
        tags=state.get("tags")
    )

    session.add(extraction)
    session.commit()
    session.close()

def register_prompt(version_id: str, field: str, template: str):
    session = Session()
    exists = session.query(Prompt).filter_by(version_id=version_id).first()
    if not exists:
        prompt = Prompt(version_id=version_id, field=field, template=template)
        session.add(prompt)
        session.commit()
    session.close()

if __name__ == "__main__":
    init_db()
    print("Tables created or updated successfully.")
