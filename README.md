# LLM Evaluation Pipeline

This repository contains a Python-based evaluation system for checking the reliability of LLM-generated responses.  
It fulfills the internship assignment requirements from the uploaded spec. :contentReference[oaicite:0]{index=0}

---

## Overview

The pipeline evaluates LLM responses in real-time across four dimensions:

- **Response Relevance & Completeness**
- **Hallucination / Factual Accuracy**
- **Latency**
- **Estimated Cost**

Inputs:
- `conversation.json` — chat conversation JSON (contains user + assistant messages)  
- `context.json` — retrieved context documents from the vector DB (used to verify factual claims)

Output:
- `out.json` (or other path provided with `-o`) — evaluation JSON with scores, flags and details.

---

## Setup Instructions (Windows / VS Code PowerShell)

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/LLM-Evaluation-Pipeline.git
cd LLM-Evaluation-Pipeline
```
2. Create and active the virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Download required NLP models/resources:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
---

## Run the evaluator
