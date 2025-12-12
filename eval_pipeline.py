#!/usr/bin/env python3
"""
eval_pipeline.py
Robust LLM response evaluation pipeline (works well on Windows/VSCode PowerShell).
Inputs: conversation.json and context.json
Outputs: evaluation_result.json (default) with relevance, completeness, hallucination, latency, cost.
"""

import argparse
import json
import time
import os
from typing import List, Tuple, Any, Dict

# Defensive imports with helpful error messages
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    import spacy
except Exception as e:
    raise RuntimeError(
        "One or more required packages are missing. Run the setup script or run:\n"
        "pip install -r requirements.txt\nThen run:\npython -m spacy download en_core_web_sm\nand ensure NLTK punkt is installed (nltk.download('punkt')).\n\nOriginal error:\n" + str(e)
    )

# Quick downloads/checks at runtime (idempotent)
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

try:
    # load spaCy model, fallback graceful error if not present
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError("spaCy model en_core_web_sm not installed. Run: python -m spacy download en_core_web_sm\nOriginal: " + str(e))

# Config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RELEVANCE_THRESHOLD = 0.45
HALLUCINATION_SIM_THRESHOLD = 0.60
COMPLETENESS_KEY_MATCH_THRESHOLD = 0.4
PRICE_PER_1K_TOKENS_USD = 0.03  # placeholder; change if needed


# ---------- Helpers ----------
def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_extract_conv(conv_json: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Extract last user message and last assistant reply from a conversation JSON.
    Works with multiple common shapes.
    """
    messages = conv_json.get("messages") or conv_json.get("chat") or conv_json
    # Normalize to list
    if isinstance(messages, dict):
        messages = [messages]
    if not isinstance(messages, list):
        # maybe it's already a list-like - cast to list
        messages = list(messages)

    last_user = ""
    last_assistant = ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("speaker") or "").lower()
        text = (m.get("text") or m.get("content") or m.get("message") or "").strip()
        if role.startswith("user"):
            last_user = text
        elif role.startswith("assistant") or role.startswith("ai") or role.startswith("bot"):
            last_assistant = text

    # fallback to direct keys
    if not last_user:
        last_user = conv_json.get("last_user_message") or conv_json.get("user") or ""
    if not last_assistant:
        last_assistant = conv_json.get("last_assistant_message") or conv_json.get("assistant") or ""

    meta = {"raw_messages_count": len(messages) if isinstance(messages, list) else 0}
    return last_user, last_assistant, meta


def flatten_context(ctx_json: Any) -> List[str]:
    """
    Extract textual context snippets from the context JSON.
    Accepts {retrieved_docs: [{text:...}, ...]} or plain list of strings.
    """
    texts = []
    if isinstance(ctx_json, dict):
        docs = ctx_json.get("retrieved_docs") or ctx_json.get("contexts") or ctx_json.get("results") or ctx_json.get("docs") or []
        if isinstance(docs, list) and docs:
            for d in docs:
                if isinstance(d, str):
                    texts.append(d)
                elif isinstance(d, dict):
                    t = d.get("text") or d.get("content") or d.get("snippet") or ""
                    if t:
                        texts.append(t)
    elif isinstance(ctx_json, list):
        for item in ctx_json:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if t:
                    texts.append(t)
    return texts


# ---------- Embedding utilities ----------
def get_embedding_model(name: str = EMBEDDING_MODEL):
    try:
        model = SentenceTransformer(name)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{name}'.\nOriginal: {e}")


def embed_texts(model, texts: List[str]):
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=float)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        # return appropriate shaped zero matrix
        return np.zeros((a.shape[0], b.shape[0] if b.size else 0))
    return cosine_similarity(a, b)


# ---------- Scoring ----------
def relevance_score(model, user_text: str, answer_text: str) -> float:
    if not user_text or not answer_text:
        return 0.0
    emb = embed_texts(model, [user_text, answer_text])
    if emb.shape[0] < 2:
        return 0.0
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    # clamp to [0,1]
    return max(0.0, min(1.0, sim))


def extract_keyphrases(text: str, max_phrases: int = 8) -> List[str]:
    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        p = chunk.text.strip().lower()
        if p and len(p.split()) <= 6:
            phrases.append(p)
    for ent in doc.ents:
        if ent.text:
            phrases.append(ent.text.lower())
    # dedupe
    out = []
    seen = set()
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= max_phrases:
            break
    return out


def completeness_score(user_text: str, answer_text: str) -> float:
    if not user_text:
        return 0.0
    keys = extract_keyphrases(user_text)
    if not keys:
        # fallback token overlap
        from nltk.tokenize import word_tokenize
        user_tokens = set(t.lower() for t in word_tokenize(user_text) if t.isalpha())
        answer_tokens = set(t.lower() for t in word_tokenize(answer_text) if t.isalpha())
        if not user_tokens:
            return 0.0
        return len(user_tokens & answer_tokens) / len(user_tokens)
    present = sum(1 for k in keys if k in answer_text.lower())
    return present / len(keys)


def hallucination_check(model, answer_text: str, context_texts: List[str], threshold: float = HALLUCINATION_SIM_THRESHOLD):
    # split into sentences
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(answer_text) if answer_text else []
    if not sentences:
        return False, []
    sent_emb = embed_texts(model, sentences)
    ctx_emb = embed_texts(model, context_texts) if context_texts else np.zeros((0, model.get_sentence_embedding_dimension()))
    sims = cosine_sim_matrix(sent_emb, ctx_emb) if ctx_emb.size else np.zeros((len(sentences), 0))
    results = []
    any_flag = False
    for i, s in enumerate(sentences):
        max_sim = float(sims[i].max()) if sims.shape[1] > 0 else 0.0
        flagged = max_sim < threshold
        any_flag = any_flag or flagged
        results.append({"sentence": s, "max_context_similarity": max_sim, "hallucinated": flagged})
    return any_flag, results


# ---------- Token & cost estimates ----------
def estimate_token_count(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # fallback heuristic: 0.75 * words
        words = text.split()
        return max(1, int(len(words) * 0.75))


def estimate_cost_usd(total_tokens: int) -> float:
    return (total_tokens / 1000.0) * PRICE_PER_1K_TOKENS_USD


# ---------- Main evaluate ----------
def evaluate(conversation_json: Dict[str, Any], context_json: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    user_text, answer_text, meta = safe_extract_conv(conversation_json)
    context_texts = flatten_context(context_json)

    model = get_embedding_model(EMBEDDING_MODEL)

    rel = relevance_score(model, user_text, answer_text)
    comp = completeness_score(user_text, answer_text)
    any_hallu, hallu_details = hallucination_check(model, answer_text, context_texts)

    latency_ms = int((time.time() - start) * 1000)

    # token estimates
    tokens_user = estimate_token_count(user_text)
    tokens_answer = estimate_token_count(answer_text)
    tokens_context = sum(estimate_token_count(t) for t in context_texts)
    total_tokens = tokens_user + tokens_answer + tokens_context
    estimated_cost = estimate_cost_usd(total_tokens)

    return {
        "relevance": rel,
        "relevance_flag": "low" if rel < RELEVANCE_THRESHOLD else "ok",
        "completeness": comp,
        "completeness_flag": "partial" if comp < COMPLETENESS_KEY_MATCH_THRESHOLD else "ok",
        "hallucination_detected": any_hallu,
        "hallucination_details": hallu_details,
        "latency_ms": latency_ms,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "meta": meta
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an LLM response against retrieved context.")
    parser.add_argument("--conversation", "-c", required=True, help="Path to conversation JSON")
    parser.add_argument("--context", "-x", required=True, help="Path to context JSON (retrieved docs)")
    parser.add_argument("--output", "-o", default="evaluation_result.json", help="Output JSON path")
    args = parser.parse_args()

    conv = load_json(args.conversation)
    ctx = load_json(args.context)

    result = evaluate(conv, ctx)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved evaluation to {args.output}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
