# Multi-Agent Cancer Registry Extraction

An open-source multi-agent LLM framework for structured cancer registry information extraction from pathology and medical reports.

## Overview

This project implements a modular multi-agent architecture for extracting clinically relevant cancer registry variables including:

- Tumor Grade
- Morphology
- TNM Staging
- Laterality
- Treatment Information

The framework is implemented using:

- LangGraph for workflow orchestration
- LangChain for prompt management and LLM interaction
- Ollama for local inference
- LLaMA 3.3 as the underlying large language model

## Architecture

The pipeline consists of sequential specialized agents:

1. Chunking Agent
2. Retriever Agent
3. Grade Extraction Agent
4. Morphology Extraction Agent
5. TNM Extraction Agent
6. Laterality Extraction Agent
7. Treatment Extraction Agent
8. Reviewer Agent
9. Evaluator Agent
10. Aggregation Agent

## Features

- Multi-agent clinical information extraction
- Structured JSON outputs
- Grounded evidence extraction
- Pipeline-level traceability
- Confidence estimation
- SQLite-based long-term memory
- Streamlit interactive interface
- Matrix calculator evaluation framework
- Local privacy-preserving inference

## Repository Structure

```text
agents/                # Specialized extraction and validation agents
workflows/             # LangGraph workflow definitions
memory/                # Long-term memory and vector store modules
mcp/                   # Message communication protocol
streamlit_app.py       # Main extraction interface
matrix_calculator_app.py # Evaluation dashboard
```

## Installation

```bash
git clone https://github.com/RihamJeeballah/multi-agent-cancer-registry-extraction.git
cd multi-agent-cancer-registry-extraction
pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

## Run the Matrix Calculator

```bash
streamlit run matrix_calculator_app.py
```

## Model

The framework uses:

- LLaMA 3.3
- Ollama local inference backend

Example:

```bash
ollama run llama3.3
```

## Applications

- Cancer registry automation
- Clinical NLP
- Pathology report structuring
- Medical information extraction
- Oncology research support

## Citation

If you use this work, please cite our upcoming publication.
