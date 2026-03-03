"""
Themia – Legal AI Copilot · FastAPI Backend
Serves the RAG + Agent pipeline over HTTP, using Ollama for LLM inference.
"""

import re
import math
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL    = "mistral"   # pulled at startup via entrypoint

# ─── Data Model ──────────────────────────────────────────────────────────────
@dataclass
class Document:
    doc_id:       str
    title:        str
    source_type:  str
    jurisdiction: str
    content:      str
    tags:         list
    date:         str
    tenant_id:    str = "firm_acme"

    @property
    def content_hash(self):
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

@dataclass
class RetrievedChunk:
    doc:          Document
    bm25_score:   float = 0.0
    dense_score:  float = 0.0
    rrf_score:    float = 0.0
    rerank_score: float = 0.0
    final_score:  float = 0.0

# ─── Corpus ──────────────────────────────────────────────────────────────────
CORPUS = [
    Document("leg_001", "EU GDPR – Article 83: Fines", "legislation", "EU",
        "Infringements shall be subject to administrative fines up to 20,000,000 EUR "
        "or 4% of the total worldwide annual turnover, whichever is higher. "
        "The supervisory authority shall ensure fines are effective, proportionate, and dissuasive. "
        "Factors include nature, gravity, duration of infringement, intentional or negligent character, "
        "and cooperation with the supervisory authority.",
        ["GDPR", "fines", "data protection", "sanctions", "compliance"], "2018-05-25"),

    Document("case_001", "CJEU – Case C-311/18 Schrems II", "case_law", "EU",
        "The Court of Justice invalidated the EU-US Privacy Shield framework. "
        "Standard Contractual Clauses (SCCs) remain valid in principle, but data controllers "
        "must conduct a Transfer Impact Assessment (TIA) to verify that the third country "
        "provides adequate protection. Data transfers to the US require supplementary measures "
        "if US surveillance law undermines SCC guarantees.",
        ["GDPR", "data transfer", "Privacy Shield", "SCCs", "TIA", "US"], "2020-07-16"),

    Document("client_001", "Merger Agreement – Acme / Globex (2023)", "client_doc", "ES",
        "Section 4.2 – Indemnification: Each party shall indemnify and hold harmless "
        "the other against losses arising from breach of representations and warranties. "
        "The aggregate liability cap is 15% of total deal value (EUR 45M). "
        "The survival period for indemnification claims is 24 months post-closing. "
        "Tax indemnities survive for the applicable statutory limitation period.",
        ["M&A", "indemnification", "representations", "warranties", "liability cap"], "2023-03-10"),

    Document("client_002", "NDA – TechStartup Inc. (Jan 2024)", "client_doc", "ES",
        "Confidential Information means all technical, financial, commercial, and legal "
        "information disclosed by either party. Obligations of confidentiality survive "
        "termination for 3 years. Exclusions apply to information already in the public domain, "
        "independently developed, or received from a third party without restriction. "
        "Breach entitles the non-breaching party to seek injunctive relief without bond.",
        ["NDA", "confidentiality", "IP", "trade secrets", "injunctive relief"], "2024-01-15"),

    Document("leg_002", "Spanish Civil Code – Article 1902: Tort Liability", "legislation", "ES",
        "He who by act or omission causes damage to another, intervening fault or negligence, "
        "is obliged to repair the damage caused. Causation must be established by the claimant. "
        "The doctrine of contributory negligence may reduce or eliminate liability proportionally. "
        "Vicarious liability arises where the tortfeasor acts under the authority of a third party.",
        ["tort", "liability", "negligence", "causation", "Spanish law", "civil code"], "1889-07-24"),

    Document("case_002", "Spanish Supreme Court – STS 3/2020: Data Breach Liability", "case_law", "ES",
        "The Supreme Court held that data controllers bear strict liability for breaches resulting "
        "from inadequate technical measures, regardless of direct fault. Compensation encompasses "
        "material damages, moral damages (daño moral), and loss of opportunity. "
        "The burden of proving adequate security measures shifts to the data controller "
        "once the plaintiff establishes the occurrence of a breach.",
        ["data breach", "liability", "GDPR", "Spanish law", "strict liability", "damages"], "2020-01-15"),
]

# ─── Retrieval helpers ────────────────────────────────────────────────────────
def bm25_score(query: str, text: str, k1=1.5, b=0.75) -> float:
    def tok(t): return re.findall(r'\b[a-z]{3,}\b', t.lower())
    qt, dt = tok(query), tok(text)
    if not qt or not dt: return 0.0
    dl, avg_dl = len(dt), 150
    tf = Counter(dt)
    score = 0.0
    for term in set(qt):
        f = tf.get(term, 0)
        if f:
            score += f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score

def tfidf_sim(query: str, text: str) -> float:
    def tok(t): return re.findall(r'\b[a-z]{3,}\b', t.lower())
    qt, dt = tok(query), tok(text)
    if not qt or not dt: return 0.0
    dl = len(dt); dtf = Counter(dt)
    vocab = set(qt) | set(dt)
    idf = {w: math.log((dl + 1) / (dtf.get(w, 0) + 1)) for w in vocab}
    qv = {w: idf.get(w, 0) for w in qt}
    dv = {w: (dtf.get(w, 0) / dl) * idf.get(w, 0) for w in vocab}
    dot = sum(qv.get(w, 0) * dv.get(w, 0) for w in vocab)
    qn = math.sqrt(sum(v**2 for v in qv.values())) or 1e-9
    dn = math.sqrt(sum(v**2 for v in dv.values())) or 1e-9
    return dot / (qn * dn)

def rrf_merge(bm25_ranked, dense_ranked, k=60) -> dict:
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_ranked, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(dense_ranked, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return scores

def cross_encoder_score(query: str, doc: Document) -> float:
    full = f"{doc.title} {doc.content} {' '.join(doc.tags)}"
    base = tfidf_sim(query, full)
    tag_boost = sum(0.05 for t in doc.tags if t.lower() in query.lower())
    year = int(doc.date[:4]) if doc.date else 2000
    return min(base + tag_boost + min((year - 2000) / 100, 0.1), 1.0)

# ─── Pipeline steps ───────────────────────────────────────────────────────────
def classify_intent(query: str) -> dict:
    q = query.lower()
    domain = "general"
    if any(w in q for w in ["gdpr","data","privacy","transfer","breach","personal"]): domain = "data_privacy"
    elif any(w in q for w in ["merger","acquisition","indemnif","warranty","deal"]): domain = "ma"
    elif any(w in q for w in ["nda","confidential","trade secret"]): domain = "nda"
    elif any(w in q for w in ["tort","liability","negligence","damage"]): domain = "tort"
    jurisdiction = "EU"
    if any(w in q for w in ["spanish","spain","tribunal supremo"]): jurisdiction = "ES"
    elif any(w in q for w in [" us ","united states","california"]): jurisdiction = "US"
    corpora = ["client_doc","legislation","case_law"]
    if domain == "ma": corpora = ["client_doc"]
    risk = "high" if any(w in q for w in ["can we","should we","is it legal","permitted","allowed"]) else "low"
    return {"domain": domain, "jurisdiction": jurisdiction, "corpora": corpora, "risk_level": risk}

def expand_query(query: str, intent: dict) -> list:
    variants = [query]
    expansions = {
        "fines": "administrative sanctions penalties",
        "fine": "administrative sanction penalty",
        "transfer": "cross-border data transfer",
        "indemnif": "indemnification hold harmless liability breach",
        "breach": "data breach security incident unauthorised access",
        "nda": "non-disclosure agreement confidentiality",
        "liability": "legal responsibility obligation to compensate",
    }
    for kw, synonyms in expansions.items():
        if kw in query.lower():
            variants.append(query + " " + synonyms.split()[0])
            break
    if intent["jurisdiction"] == "ES": variants.append(query + " under Spanish law")
    elif intent["jurisdiction"] == "EU": variants.append(query + " under EU regulation")
    return variants

def retrieve(query: str, intent: dict, variants: list, top_k=6) -> list:
    relevant_types = intent["corpora"]
    tenant_corpus = [d for d in CORPUS if d.source_type in relevant_types]
    doc_map = {d.doc_id: d for d in tenant_corpus}
    combined_rrf = {}
    bm25_best, dense_best = {}, {}

    for v in variants:
        bm25_raw   = [(d.doc_id, bm25_score(v, d.content + " " + d.title)) for d in tenant_corpus]
        dense_raw  = [(d.doc_id, tfidf_sim(v, d.content + " " + d.title)) for d in tenant_corpus]
        bm25_sort  = sorted(bm25_raw,  key=lambda x: x[1], reverse=True)
        dense_sort = sorted(dense_raw, key=lambda x: x[1], reverse=True)
        for did, s in bm25_raw:  bm25_best[did]  = max(bm25_best.get(did, 0), s)
        for did, s in dense_raw: dense_best[did] = max(dense_best.get(did, 0), s)
        for did, s in rrf_merge(bm25_sort, dense_sort).items():
            combined_rrf[did] = combined_rrf.get(did, 0) + s

    ranked = sorted(combined_rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(doc=doc_map[did],
                       bm25_score=round(bm25_best.get(did,0),4),
                       dense_score=round(dense_best.get(did,0),4),
                       rrf_score=round(s,6))
        for did, s in ranked if did in doc_map
    ]

def rerank_and_dedup(query: str, chunks: list, top_n=4) -> list:
    for c in chunks:
        c.rerank_score = round(cross_encoder_score(query, c.doc), 4)
    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    seen, result = set(), []
    def jaccard(a, b):
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if (ta | tb) else 0
    for c in ranked:
        h = c.doc.content_hash
        if h in seen: continue
        if any(jaccard(c.doc.content, r.doc.content) > 0.85 for r in result): continue
        seen.add(h); result.append(c)
    return result[:top_n]

async def call_ollama(system_prompt: str, user_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

def verify_citations(answer: str, final_chunks: list) -> dict:
    n = len(final_chunks)
    used = set(int(c) for c in re.findall(r'\[(\d+)\]', answer))
    invalid = [c for c in used if c > n or c < 1]
    flags = []
    if invalid:
        flags.append(f"Out-of-range citation indices: {invalid} (only {n} sources available)")
    legal_kw = ["shall","must","liable","obligation","prohibited","permitted","sanction","fine","penalty"]
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    uncited = [s.strip() for s in sentences
               if any(k in s.lower() for k in legal_kw)
               and not re.search(r'\[\d+\]', s)
               and len(s) > 40 and not s.startswith("#")]
    if uncited:
        flags.append(f"{len(uncited)} potentially uncited legal claim(s) found")
    return {"valid": len(invalid) == 0, "flags": flags, "citations_used": sorted(used)}

# ─── API ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Themia API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    tenant_id: str = "firm_acme"

@app.get("/health")
async def health():
    # Check Ollama is reachable
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = any(OLLAMA_MODEL in m for m in models)
        return {"status": "ok", "ollama": "connected", "model_ready": model_ready, "model": OLLAMA_MODEL}
    except Exception as e:
        return {"status": "ok", "ollama": "unreachable", "error": str(e)}

@app.post("/query")
async def run_query(req: QueryRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 – Intent
    intent = classify_intent(query)

    # Step 2 – Expand
    variants = expand_query(query, intent)

    # Step 3 – Retrieve
    retrieved = retrieve(query, intent, variants)
    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant documents found in corpus")

    # Step 4 – Rerank
    final_chunks = rerank_and_dedup(query, retrieved)

    # Step 5 – LLM
    context_block = ""
    for i, c in enumerate(final_chunks, 1):
        context_block += (
            f"[{i}] SOURCE: {c.doc.title}\n"
            f"    Type: {c.doc.source_type} | Jurisdiction: {c.doc.jurisdiction} | Date: {c.doc.date}\n"
            f"    Content: {c.doc.content}\n\n"
        )

    system_prompt = """You are Themia, an AI legal copilot for law firms.
Answer legal questions STRICTLY grounded in the provided context.
Rules:
1. Reason step by step before concluding.
2. Tag every factual claim with [n] matching the source index.
3. If the context is insufficient, say so explicitly. Do not invent.
4. Structure: ## Reasoning / ## Answer / ## Caveats
5. End with: "⚖ This analysis is based solely on retrieved documents and does not constitute legal advice." """

    user_prompt = f"RETRIEVED LEGAL CONTEXT:\n{context_block}\nLEGAL QUESTION: {query}\n\nDomain: {intent['domain']} | Jurisdiction: {intent['jurisdiction']}"

    try:
        answer = await call_ollama(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    # Step 6 – Verify
    citation_check = verify_citations(answer, final_chunks)

    return {
        "query": query,
        "intent": intent,
        "expanded_queries": variants,
        "retrieved_chunks": [
            {
                "index": i + 1,
                "doc_id": c.doc.doc_id,
                "title": c.doc.title,
                "source_type": c.doc.source_type,
                "jurisdiction": c.doc.jurisdiction,
                "date": c.doc.date,
                "content": c.doc.content,
                "tags": c.doc.tags,
                "bm25_score": c.bm25_score,
                "dense_score": c.dense_score,
                "rrf_score": c.rrf_score,
                "rerank_score": c.rerank_score,
            }
            for i, c in enumerate(final_chunks)
        ],
        "answer": answer,
        "citation_check": citation_check,
    }
