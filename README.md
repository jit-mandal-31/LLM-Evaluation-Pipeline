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
Run the script with sample inputs:
```bash
python eval_pipeline.py -c sample_data/conversation.json -x sample_data/context.json -o out.json
```
- The script prints the evaluation and saves results to out.json.

- If out.json is created, the pipeline executed successfully.

## File descriptions
---
```bash
LLM-Evaluation-Pipeline/
│
├── eval_pipeline.py        # Main evaluation script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── setup_env.ps1           # Windows setup helper (PowerShell)
├── sample_data/
│   ├── conversation.json   # Example conversation input
│   └── context.json        # Example retrieved docs input
└── .gitignore
```
## Pipeline architecture (high-level)
---
### **1. Input Loader**
- Load conversation.json and context.json, extract last user message and assistant response.
  
### **2. Preprocessing**

- Clean and normalize texts; extract retrieved context snippets.
  
### **3. Embedding layer**
- Use Sentence-BERT (all-MiniLM-L6-v2) to create embeddings for:
  
    - user message
      
    - assistant response (and its sentences)
      
    - retrieved context chunks
      
### **4. Relevance check**

- Compute cosine similarity between user message and entire assistant response.
  
- Report a relevance score (0–1) and a simple flag (ok / low)
  
### **5. Completeness check**

- Extract keyphrases (spaCy noun-chunks + named entities) from the question.
  
- Check fraction present in the assistant response.
  
- Provide completeness score and flag (ok / partial).
  
### **6. Hallucination / factual check**

- Split assistant response into sentences.
  
- For each sentence, compute max cosine similarity to any context chunk.

- If max similarity < threshold (default 0.60) → mark sentence as hallucinated.
  
- Provide sentence-level details for human inspection.
  
### **7. Latency & cost estimation**

- Measure wall-clock latency (ms) for evaluation.
  
- Estimate token usage using tiktoken if available; fallback heuristic otherwise.
  
- Calculate an estimated cost (configurable price per 1k tokens).
  
### **8. Output**

- Compose JSON containing scores, flags, hallucination details, latency, estimated tokens and cost.
--- 
## Why this design?

**Cost-efficient** — *uses a small embedding model (no LLM API calls) so evaluation is inexpensive.*

**Fast & real-time friendly** — *embeddings+cosine similarity are lightweight and parallelizable.*

**Interpretable** — *sentence-level hallucination flags give actionable signals for reviewers or automated moderation.*

**Modular** — *components (embedding model, keyphrase extractor, thresholds) are easily swappable.*
---
## How this scales

### 1. Persistent service mode 

- Run the evaluator as a persistent service (load model once, handle many requests).

### 2. Batching & matrix ops
- Batch multiple texts into a single embedding call and process similarities using matrix multiplication for HPC efficiency.
### 3. Caching
- Cache context embeddings (frequently retrieved docs) to avoid repeated embedding costs.
### 4. Adaptive checks
- Run lightweight checks for every response; escalate to heavier checks (LLM-based verification) only when needed.
### 5. Autoscaling
- Use worker pools or serverless functions with autoscaling for peak loads.
