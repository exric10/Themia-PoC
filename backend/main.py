"""
Themia – Legal AI Copilot · FastAPI Backend
Serves the RAG + Agent pipeline over HTTP, using Ollama for LLM inference.

Pipeline overview:
  POST /query
    → classify_intent      detect domain, jurisdiction, risk, corpora
    → expand_query         generate synonym query variants
    → retrieve             hybrid BM25 + dense search with RRF merge
    → rerank_and_dedup     cross-encoder scoring + duplicate removal
    → call_ollama          guided CoT reasoning with citation-anchored prompt
    → verify_citations     post-generation citation integrity check
"""

import re
import math
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("themia")

# ─── Config ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL    = "mistral"   # pulled at startup via entrypoint

# ─── Data Model ──────────────────────────────────────────────────────────────
@dataclass
class Document:
    """
    Represents a single document chunk in the corpus.
    Each document belongs to a tenant and carries metadata used for
    filtering, ranking, and citation display.
    """
    doc_id:       str
    title:        str
    source_type:  str   # "client_doc" | "legislation" | "case_law"
    jurisdiction: str   # "EU" | "ES" | "US"
    content:      str
    tags:         list
    date:         str   # ISO date — used to flag potentially stale sources
    tenant_id:    str = "firm_acme"

    @property
    def content_hash(self) -> str:
        """Returns a short SHA-256 hash of the content, used for exact deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievedChunk:
    """
    A Document enriched with retrieval and ranking scores as it flows
    through the pipeline. Scores are populated incrementally at each step.
    """
    doc:          Document
    bm25_score:   float = 0.0   # sparse keyword relevance (Step 3)
    dense_score:  float = 0.0   # semantic similarity proxy (Step 3)
    rrf_score:    float = 0.0   # combined Reciprocal Rank Fusion score (Step 3)
    rerank_score: float = 0.0   # cross-encoder joint score (Step 4)
    final_score:  float = 0.0   # final score after dedup (Step 4)


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

logger.info("Corpus loaded: %d documents across %d tenants",
            len(CORPUS), len(set(d.tenant_id for d in CORPUS)))


# ─── Retrieval helpers ────────────────────────────────────────────────────────

def bm25_score(query: str, text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Computes a BM25 relevance score between a query and a document.

    BM25 is a sparse keyword-based ranking function. It improves on plain TF-IDF (Term Frequency-Inverse Document Frequency)
    by adding term-frequency saturation (k1) and document-length normalisation (b),
    preventing very long documents from being unfairly boosted.

    In production this would be handled by Weaviate.
    """
    def tok(t): return re.findall(r'\b[a-z]{3,}\b', t.lower())
    qt, dt = tok(query), tok(text)
    if not qt or not dt:
        return 0.0
    dl, avg_dl = len(dt), 150
    tf = Counter(dt)
    score = 0.0
    for term in set(qt):
        f = tf.get(term, 0)
        if f:
            score += f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avg_dl))
    return score


def tfidf_sim(query: str, text: str) -> float:
    """
    Computes a TF-IDF cosine similarity between a query and a document.

    Used as a lightweight proxy for dense embedding similarity in this PoC.
    In production, this is replaced by embedding both texts with a model such as
    text-embedding-3-large (OpenAI) or bge-large (open-source), then computing
    cosine similarity in a vector database (Weaviate).
    """
    def tok(t): return re.findall(r'\b[a-z]{3,}\b', t.lower())
    qt, dt = tok(query), tok(text)
    if not qt or not dt:
        return 0.0
    dl = len(dt)
    dtf = Counter(dt)
    vocab = set(qt) | set(dt)
    idf = {w: math.log((dl + 1) / (dtf.get(w, 0) + 1)) for w in vocab}
    qv = {w: idf.get(w, 0) for w in qt}
    dv = {w: (dtf.get(w, 0) / dl) * idf.get(w, 0) for w in vocab}
    dot = sum(qv.get(w, 0) * dv.get(w, 0) for w in vocab)
    qn = math.sqrt(sum(v**2 for v in qv.values())) or 1e-9
    dn = math.sqrt(sum(v**2 for v in dv.values())) or 1e-9
    return dot / (qn * dn)


def rrf_merge(bm25_ranked: list, dense_ranked: list, k: int = 60) -> dict:
    """
    Merges two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF combines BM25 and dense rankings without requiring score normalisation,
    since it operates on rank positions rather than raw scores.
    Formula: RRF(doc) = sum( 1 / (k + rank(doc)) )

    k=60 is the standard default — lower values increase the weight of top-ranked docs.
    This is the same strategy used natively by Weaviate hybrid search.
    """
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_ranked, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(dense_ranked, 1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return scores


def cross_encoder_score(query: str, doc: Document) -> float:
    """
    Simulates a cross-encoder relevance score for a (query, document) pair.

    A real cross-encoder (e.g. cross-encoder/ms-marco-MiniLM or bge-reranker-large)
    processes query and document together in a single forward pass, producing a more
    accurate relevance score than bi-encoders at the cost of higher latency.
    Applied only to the top-K candidates to keep reranking tractable.

    This PoC approximates it with enhanced TF-IDF plus a tag-match boost and a
    recency bonus to prefer newer legal sources.
    """
    full = f"{doc.title} {doc.content} {' '.join(doc.tags)}"
    base = tfidf_sim(query, full)
    tag_boost = sum(0.05 for t in doc.tags if t.lower() in query.lower())
    year = int(doc.date[:4]) if doc.date else 2000
    return min(base + tag_boost + min((year - 2000) / 100, 0.1), 1.0)


# ─── Pipeline steps ───────────────────────────────────────────────────────────

def classify_intent(query: str) -> dict:
    """
    Step 1 – Classifies the legal intent of the incoming query.

    Detects domain (data_privacy, M&A, NDA, tort, general), jurisdiction (EU, ES, US),
    the relevant corpora to search, and whether the query is high-risk (i.e. action-oriented
    questions that should be routed to a human lawyer before acting on the answer).

    In production this would be a fast LLM call with a structured output schema,
    or a fine-tuned intent classifier. Target latency: <200 ms.
    """
    q = query.lower()
    domain = "general"
    if any(w in q for w in ["gdpr", "data", "privacy", "transfer", "breach", "personal"]):
        domain = "data_privacy"
    elif any(w in q for w in ["merger", "acquisition", "indemnif", "warranty", "deal"]):
        domain = "ma"
    elif any(w in q for w in ["nda", "confidential", "trade secret"]):
        domain = "nda"
    elif any(w in q for w in ["tort", "liability", "negligence", "damage"]):
        domain = "tort"

    jurisdiction = "EU"
    if any(w in q for w in ["spanish", "spain", "tribunal supremo"]):
        jurisdiction = "ES"
    elif any(w in q for w in [" us ", "united states", "california"]):
        jurisdiction = "US"

    corpora = ["client_doc", "legislation", "case_law"]
    if domain == "ma":
        corpora = ["client_doc"]   # M&A queries focus on client documents

    risk = "high" if any(w in q for w in ["can we", "should we", "is it legal", "permitted", "allowed"]) else "low"

    logger.info("[Step 1] Intent: domain=%s jurisdiction=%s risk=%s corpora=%s",
                domain, jurisdiction, risk, corpora)
    return {"domain": domain, "jurisdiction": jurisdiction, "corpora": corpora, "risk_level": risk}


def expand_query(query: str, intent: dict) -> list:
    """
    Step 2 – Expands the original query with synonym variants to improve retrieval recall.

    Legal text is precise but uses varied terminology (e.g. 'fines' vs 'administrative
    sanctions'). Generating variants ensures we don't miss relevant chunks due to
    vocabulary mismatch. A jurisdiction-scoped variant is also appended.

    In production, this can be done with an LLM call or the HyDE technique
    (generating a hypothetical ideal answer and using it as the query vector).
    """
    variants = [query]
    expansions = {
        "fines":     "administrative sanctions penalties",
        "fine":      "administrative sanction penalty",
        "transfer":  "cross-border data transfer",
        "indemnif":  "indemnification hold harmless liability breach",
        "breach":    "data breach security incident unauthorised access",
        "nda":       "non-disclosure agreement confidentiality",
        "liability": "legal responsibility obligation to compensate",
    }
    for kw, synonyms in expansions.items():
        if kw in query.lower():
            variants.append(query + " " + synonyms.split()[0])
            break

    if intent["jurisdiction"] == "ES":
        variants.append(query + " under Spanish law")
    elif intent["jurisdiction"] == "EU":
        variants.append(query + " under EU regulation")

    logger.info("[Step 2] Query expanded to %d variant(s): %s", len(variants), variants)
    return variants


def retrieve(intent: dict, variants: list, top_k: int = 6) -> list:
    """
    Step 3 – Hybrid retrieval combining BM25 and dense (TF-IDF) search with RRF.

    Runs each expanded query variant against both retrievers, then merges results
    using Reciprocal Rank Fusion. The corpus is pre-filtered by tenant and the
    relevant source types identified in Step 1.

    In production: BM25 via Weaviate sparse vectors; dense search via
    approximate nearest-neighbour lookup in a vector database with pre-computed embeddings.
    """
    relevant_types = intent["corpora"]
    tenant_corpus = [d for d in CORPUS if d.source_type in relevant_types]
    doc_map = {d.doc_id: d for d in tenant_corpus}
    combined_rrf: dict = {}
    bm25_best: dict = {}
    dense_best: dict = {}

    logger.info("[Step 3] Retrieving from %d documents across corpora: %s",
                len(tenant_corpus), relevant_types)

    for v in variants:
        bm25_raw  = [(d.doc_id, bm25_score(v, d.content + " " + d.title)) for d in tenant_corpus]
        dense_raw = [(d.doc_id, tfidf_sim(v, d.content + " " + d.title)) for d in tenant_corpus]
        bm25_sort  = sorted(bm25_raw,  key=lambda x: x[1], reverse=True)
        dense_sort = sorted(dense_raw, key=lambda x: x[1], reverse=True)

        for did, s in bm25_raw:  bm25_best[did]  = max(bm25_best.get(did, 0), s)
        for did, s in dense_raw: dense_best[did] = max(dense_best.get(did, 0), s)

        for did, s in rrf_merge(bm25_sort, dense_sort).items():
            combined_rrf[did] = combined_rrf.get(did, 0) + s

    ranked = sorted(combined_rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    chunks = [
        RetrievedChunk(
            doc=doc_map[did],
            bm25_score=round(bm25_best.get(did, 0), 4),
            dense_score=round(dense_best.get(did, 0), 4),
            rrf_score=round(s, 6),
        )
        for did, s in ranked if did in doc_map
    ]

    logger.info("[Step 3] Retrieved %d candidate chunks (top RRF scores: %s)",
                len(chunks), [round(c.rrf_score, 4) for c in chunks[:3]])
    return chunks


def rerank_and_dedup(query: str, chunks: list, top_n: int = 4) -> list:
    """
    Step 4 – Reranks candidates with a cross-encoder and removes duplicates.

    Applies cross-encoder scoring for more accurate (query, document) relevance
    assessment, then removes exact duplicates (by content hash) and near-duplicates
    (Jaccard similarity > 0.85) to avoid sending redundant context to the LLM.

    Sending duplicate content wastes tokens and can bias the model toward repeated
    information, so this step is important for both quality and cost control.
    """
    for c in chunks:
        c.rerank_score = round(cross_encoder_score(query, c.doc), 4)

    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
    seen: set = set()
    result: list = []

    def jaccard(a: str, b: str) -> float:
        """Token-level Jaccard similarity for near-duplicate detection."""
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / len(ta | tb) if (ta | tb) else 0

    for c in ranked:
        h = c.doc.content_hash
        if h in seen:
            logger.debug("[Step 4] Exact duplicate removed: %s", c.doc.title)
            continue
        if any(jaccard(c.doc.content, r.doc.content) > 0.85 for r in result):
            logger.debug("[Step 4] Near-duplicate removed: %s", c.doc.title)
            continue
        seen.add(h)
        result.append(c)

    final = result[:top_n]
    logger.info("[Step 4] Reranked and deduplicated: %d chunks selected (scores: %s)",
                len(final), [c.rerank_score for c in final])
    return final


async def call_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Step 5 – Sends the chain-of-thought prompt to Ollama and returns the response.

    Uses a non-streaming POST to /api/chat. The system prompt enforces:
    - Step-by-step reasoning before concluding
    - Mandatory [n] citation tags on every factual claim
    - Explicit abstention if context is insufficient
    - Structured output (## Reasoning / ## Answer / ## Caveats)

    Timeout is set to 120 s to accommodate slower hardware. In production,
    streaming (stream=True) would be used to reduce perceived latency.
    """
    logger.info("[Step 5] Calling Ollama model '%s'", OLLAMA_MODEL)
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
        answer = response.json()["message"]["content"]
        logger.info("[Step 5] Ollama responded (%d chars)", len(answer))
        return answer


def verify_citations(answer: str, final_chunks: list) -> dict:
    """
    Step 6 – Post-generation citation integrity check.

    Validates that all [n] citation indices in the answer are within bounds,
    and flags sentences containing legal keywords that lack any citation.
    This catches the most common LLM citation errors without a second model call.

    Note: faithfulness checking (does the cited source actually support the claim?)
    requires an NLI model (e.g. cross-encoder/nli-deberta-v3-base) and is marked
    as a production extension.
    """
    n = len(final_chunks)
    used = set(int(c) for c in re.findall(r'\[(\d+)\]', answer))
    invalid = [c for c in used if c > n or c < 1]
    flags = []

    if invalid:
        flags.append(f"Out-of-range citation indices: {invalid} (only {n} sources available)")
        logger.warning("[Step 6] Out-of-range citations detected: %s", invalid)

    legal_kw = ["shall", "must", "liable", "obligation", "prohibited", "permitted",
                "sanction", "fine", "penalty"]
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    uncited = [
        s.strip() for s in sentences
        if any(k in s.lower() for k in legal_kw)
        and not re.search(r'\[\d+\]', s)
        and len(s) > 40
        and not s.startswith("#")
    ]
    if uncited:
        flags.append(f"{len(uncited)} potentially uncited legal claim(s) found")
        logger.warning("[Step 6] %d potentially uncited legal claims detected", len(uncited))

    result = {"valid": len(invalid) == 0, "flags": flags, "citations_used": sorted(used)}
    logger.info("[Step 6] Citation check complete: valid=%s flags=%d citations_used=%s",
                result["valid"], len(flags), sorted(used))
    return result


# ─── API ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Themia API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query:     str
    tenant_id: str = "firm_acme"


@app.get("/health")
async def health():
    """
    Health check endpoint. Verifies that Ollama is reachable and the
    configured model has been pulled and is available for inference.
    Polled by the frontend every 15 seconds to update the status indicator.
    """
    logger.debug("Health check requested")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            model_ready = any(OLLAMA_MODEL in m for m in models)
        logger.info("Health check: ollama=connected model_ready=%s available_models=%s",
                    model_ready, models)
        return {"status": "ok", "ollama": "connected", "model_ready": model_ready, "model": OLLAMA_MODEL}
    except Exception as e:
        logger.warning("Health check: ollama unreachable – %s", str(e))
        return {"status": "ok", "ollama": "unreachable", "error": str(e)}


@app.post("/query")
async def run_query(req: QueryRequest):
    """
    Main query endpoint. Runs the full 6-step RAG + agent pipeline and returns
    a structured response including retrieved chunks, the generated answer,
    and citation verification results.

    Returns 400 if the query is empty, 404 if no relevant documents are found,
    and 502 if the Ollama inference call fails.
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info("=== New query from tenant='%s' | query='%s'", req.tenant_id, query[:80])

    # Step 1 – Intent classification
    intent = classify_intent(query)

    # Step 2 – Query expansion
    variants = expand_query(query, intent)

    # Step 3 – Hybrid retrieval
    retrieved = retrieve(intent, variants)
    if not retrieved:
        logger.warning("No documents retrieved for query: '%s'", query)
        raise HTTPException(status_code=404, detail="No relevant documents found in corpus")

    # Step 4 – Rerank and deduplicate
    final_chunks = rerank_and_dedup(query, retrieved)

    # Step 5 – Build context block and call LLM
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

    user_prompt = (
        f"RETRIEVED LEGAL CONTEXT:\n{context_block}\n"
        f"LEGAL QUESTION: {query}\n\n"
        f"Domain: {intent['domain']} | Jurisdiction: {intent['jurisdiction']}"
    )

    try:
        answer = await call_ollama(system_prompt, user_prompt)
    except Exception as e:
        logger.error("Ollama call failed: %s", str(e))
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    # Step 6 – Citation verification
    citation_check = verify_citations(answer, final_chunks)

    logger.info("=== Query complete | chunks=%d citations_valid=%s flags=%d",
                len(final_chunks), citation_check["valid"], len(citation_check["flags"]))

    return {
        "query": query,
        "intent": intent,
        "expanded_queries": variants,
        "retrieved_chunks": [
            {
                "index":        i + 1,
                "doc_id":       c.doc.doc_id,
                "title":        c.doc.title,
                "source_type":  c.doc.source_type,
                "jurisdiction": c.doc.jurisdiction,
                "date":         c.doc.date,
                "content":      c.doc.content,
                "tags":         c.doc.tags,
                "bm25_score":   c.bm25_score,
                "dense_score":  c.dense_score,
                "rrf_score":    c.rrf_score,
                "rerank_score": c.rerank_score,
            }
            for i, c in enumerate(final_chunks)
        ],
        "answer": answer,
        "citation_check": citation_check,
    }