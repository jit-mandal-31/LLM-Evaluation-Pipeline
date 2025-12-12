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
