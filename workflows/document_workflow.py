from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from langchain_core.runnables.retry import RunnableRetry
from langchain_core.runnables import RunnableLambda

from agents.chunking_agent_1 import chunking_agent
from agents.retriever_agent_2 import retriever_agent
from agents.grade_extraction_agent_3 import grade_extractor
from agents.morph_extraction_agent_4 import morph_extractor
from agents.tnm_extraction_agent_5 import tnm_extractor
from agents.laterality_extraction_agent_6 import laterality_extractor
from agents.treatment_extraction_agent_7 import treatment_extractor
from agents.reviewer_agent_8 import review_agent
from agents.evaluator_agent_9 import evaluator_agent
from agents.aggregation_agent_10 import aggregation_agent

class State(TypedDict):
    report: str
    question: str
    chunks: list
    retrieved_chunks: dict
    vectorstore: Any

    grade: str
    morphology: str
    tnm: str
    laterality: str
    treatment: str

    grade_valid: bool
    morphology_valid: bool
    tnm_valid: bool
    laterality_valid: bool
    treatment_valid: bool

    grade_confidence: float
    morphology_confidence: float
    tnm_confidence: float
    laterality_confidence: float
    treatment_confidence: float

    grade_correct: bool
    morphology_correct: bool
    tnm_correct: bool
    laterality_correct: bool
    treatment_correct: bool

    summary: str
    mcp_log: list

def build_workflow():
    graph = StateGraph(State)

  

    # Add nodes (retry-wrapped where needed)
    graph.add_node("chunker", chunking_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("grade_extractor", grade_extractor)
    graph.add_node("morph_extractor", morph_extractor)
    graph.add_node("tnm_extractor", tnm_extractor)
    graph.add_node("laterality_extractor", laterality_extractor)
    graph.add_node("treatment_extractor", treatment_extractor)
    graph.add_node("review_agent", review_agent)
    graph.add_node("evaluator_agent", evaluator_agent)
    graph.add_node("aggregation_agent", aggregation_agent)

    # Set entry point and edges
    graph.set_entry_point("chunker")
    graph.add_edge("chunker", "retriever")
    graph.add_edge("retriever", "grade_extractor")
    graph.add_edge("grade_extractor", "morph_extractor")
    graph.add_edge("morph_extractor", "tnm_extractor")
    graph.add_edge("tnm_extractor", "laterality_extractor")
    graph.add_edge("laterality_extractor", "treatment_extractor")
    graph.add_edge("treatment_extractor", "review_agent")
    graph.add_edge("review_agent", "evaluator_agent")
    graph.add_edge("evaluator_agent", "aggregation_agent")
    graph.add_edge("aggregation_agent", END)

    compiled = graph.compile()
    return compiled, graph
