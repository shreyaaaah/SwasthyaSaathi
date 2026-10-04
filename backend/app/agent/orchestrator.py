import os
import re
import sys
import json
import time
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure stdout and stderr handle UTF-8 characters on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.dirname(APP_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from groq import Groq
from app.config import settings
from app.db import SessionLocal, init_db
from app.models import SymptomLog
from app.rag.retriever import get_retriever

# Thread pool for asynchronous non-blocking DB logging
bg_executor = ThreadPoolExecutor(max_workers=5)

# ===========================================================================
# MODEL CHOICE EXPLANATION
# ===========================================================================
# We use `qwen/qwen3.8-27b` as primary, falling back to `openai/gpt-oss-120b`
# then `openai/gpt-oss-20b`. groq/compound is excluded from all user-facing
# and audit chains. The exact model string is printed at orchestrator start.
# ===========================================================================

# ---------------------------------------------------------------------------
# Circuit Breaker: per-model 429 cooldown
# ---------------------------------------------------------------------------

_circuit_breaker_lock = threading.Lock()
_model_cooldown_until: Dict[str, float] = {}   # model_id -> epoch seconds
CIRCUIT_BREAKER_COOLDOWN_SECS = 180   # 3-minute default cooldown

def _mark_model_429(model_id: str, retry_after_secs: Optional[int] = None):
    """Record a 429 hit for model_id. Respects Retry-After header if present."""
    cooldown = float(retry_after_secs) if retry_after_secs else float(CIRCUIT_BREAKER_COOLDOWN_SECS)
    until = time.time() + cooldown
    with _circuit_breaker_lock:
        _model_cooldown_until[model_id] = until
    print(f"  [CIRCUIT BREAKER] Model '{model_id}' cooling down for {cooldown:.0f}s (until {time.strftime('%H:%M:%S', time.localtime(until))})")

def _model_is_cooling(model_id: str) -> bool:
    with _circuit_breaker_lock:
        until = _model_cooldown_until.get(model_id, 0.0)
    return time.time() < until

def _available_models_in_order(models_to_try: List[str]) -> List[str]:
    """Return models that are not currently cooling down, in order."""
    return [m for m in models_to_try if not _model_is_cooling(m)]


# ---------------------------------------------------------------------------
# Cross-Encoder Specificity Gate
# Threshold: CE < 0  → skip KB and sufficiency check, go straight to web search
#            CE >= 0 → run sufficiency check in parallel with web search
# The class overlap analysis (newborn vaccines OUT-KB at 1.95, influenza IN-KB
# at 3.80; leptospirosis OUT-KB at 4.27) means no midpoint threshold is valid.
# Instead: CE < 0 is a hard skip; everything else goes through sufficiency check.
# ---------------------------------------------------------------------------
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("  [CE-GATE] Cross-encoder loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"  [CE-GATE] Cross-encoder unavailable: {e}.")
            _cross_encoder = None
    return _cross_encoder

CE_SKIP_THRESHOLD = 0.0   # Max-CE < 0 → skip KB entirely, no sufficiency check


def specificity_gate(query: str, raw_chunks: List[Dict[str, Any]], faiss_floor: float = 0.45,
                     encode_query_str: str = None) -> Dict[str, Any]:
    """
    Two-stage specificity gate:
    Stage 1 — FAISS floor: drop all chunks with cosine score < faiss_floor.
    Stage 2 — Cross-encoder rerank: score (query, chunk) for ALL valid chunks.
              If max CE score < 0: KB lacks specificity completely → skip KB and sufficiency.
              If max CE score >= 0: KB partially relevant → run sufficiency check.
              If CE model is unavailable, fails closed (passed=False).

    Prints:
      - The exact string passed to model.encode() for FAISS embedding (encode_query_str)
      - The exact (query, chunk) pair strings passed to CE.predict()
    """
    # Print the exact string passed to encode() for FAISS embedding
    embed_str = encode_query_str if encode_query_str is not None else query
    print(f"  [FAISS ENCODE STRING] Exact string passed to model.encode(): {repr(embed_str)}")

    valid = [c for c in raw_chunks if c.get("score", 0.0) >= faiss_floor]
    if not valid:
        return {
            "passed": False,
            "skip_kb": True,
            "reason": f"0/{len(raw_chunks)} chunks above FAISS floor {faiss_floor}",
            "faiss_top1": raw_chunks[0]["score"] if raw_chunks else 0.0,
            "max_ce_score": None,
            "chunk_ce_scores": [],
            "valid_chunks": []
        }

    faiss_top1 = valid[0].get("score", 0.0)
    ce = _get_cross_encoder()
    if ce is None:
        return {
            "passed": False,
            "skip_kb": True,
            "reason": f"FAISS floor passed ({faiss_top1:.4f} >= {faiss_floor}); CE model unavailable → fail closed",
            "faiss_top1": faiss_top1,
            "max_ce_score": None,
            "chunk_ce_scores": [],
            "valid_chunks": valid
        }

    try:
        pairs = [(query, c.get("chunk_text", c.get("snippet", ""))[:512]) for c in valid]
        # Print the exact strings passed to CE.predict()
        for i, (q_str, chunk_str) in enumerate(pairs):
            print(f"  [CE PAIR {i}] query={repr(q_str[:80])} | chunk={repr(chunk_str[:80])}")
        scores = ce.predict(pairs)
        chunk_ce_scores = []
        for idx, (c, sc) in enumerate(zip(valid, scores)):
            chunk_ce_scores.append({
                "chunk_id": c.get("chunk_id", idx),
                "doc_name": c.get("doc_name"),
                "section": c.get("section"),
                "faiss_score": float(c.get("score", 0.0)),
                "ce_score": float(sc),
                "snippet_80": c.get("chunk_text", c.get("snippet", ""))[:80]
            })
        max_ce_score = float(max(scores))
    except Exception as e:
        print(f"  [CE-GATE] CE predict error: {e}. Failing closed to trigger fallback.")
        return {
            "passed": False,
            "skip_kb": True,
            "reason": f"FAISS floor passed ({faiss_top1:.4f} >= {faiss_floor}); CE predict error ({e}) → fail closed",
            "faiss_top1": faiss_top1,
            "max_ce_score": None,
            "chunk_ce_scores": [],
            "valid_chunks": valid
        }

    # CE < 0: skip KB entirely (hard skip - no sufficiency check)
    skip_kb = max_ce_score < CE_SKIP_THRESHOLD
    # CE >= 0: run sufficiency check, may still fallback
    passed = not skip_kb

    reason = (
        f"Max-CE={max_ce_score:.4f} < 0 — KB entirely off-topic → skip KB, go straight to web"
        if skip_kb else
        f"Max-CE={max_ce_score:.4f} >= 0 — KB partially relevant → run sufficiency check"
    )
    print(f"  [CE-GATE] FAISS top1={faiss_top1:.4f} | Max-CE={max_ce_score:.4f} | passed={passed} | skip_kb={skip_kb}")
    return {
        "passed": passed,
        "skip_kb": skip_kb,
        "reason": reason,
        "faiss_top1": faiss_top1,
        "max_ce_score": max_ce_score,
        "chunk_ce_scores": chunk_ce_scores,
        "valid_chunks": valid
    }


def sufficiency_check(client: Groq, active_model: str, query: str, valid_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates whether retrieved KB chunks contain all necessary facts to answer query.
    Small JSON LLM call. Returns: {"is_sufficient": bool, "missing_facts": List[str], "reason": str}
    """
    if not valid_chunks:
        return {"is_sufficient": False, "missing_facts": ["No valid KB chunks"], "reason": "No valid chunks"}

    summaries = []
    for idx, c in enumerate(valid_chunks, 1):
        doc = c.get("doc_name", "Doc")
        sec = c.get("section", "Section")
        text = c.get("chunk_text", c.get("snippet", ""))[:400]
        summaries.append(f"Chunk [{idx}] ({doc} - {sec}):\n{text}")

    prompt = (
        f"Query: \"{query}\"\n\n"
        f"Retrieved KB Chunks:\n" + "\n\n".join(summaries) + "\n\n"
        "Evaluate if these retrieved health guideline chunks contain the SPECIFIC clinical/medical facts "
        "needed to directly and accurately answer the user query.\n"
        "Respond ONLY with valid JSON in this exact structure:\n"
        '{"is_sufficient": true/false, "missing_facts": ["<fact>", "..."], "reason": "<one sentence>"}'
    )

    try:
        res = safe_chat_completion(
            client,
            active_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800
        )
        raw = res.choices[0].message.content or ""
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = re.sub(r"^```[a-zA-Z]*\n?", "", raw_clean)
            raw_clean = re.sub(r"\n?```$", "", raw_clean).strip()
        match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
        if match:
            raw_clean = match.group(0)

        data = json.loads(raw_clean)
        return {
            "is_sufficient": bool(data.get("is_sufficient", False)),
            "missing_facts": data.get("missing_facts", []),
            "reason": data.get("reason", "Sufficiency check complete")
        }
    except Exception as e:
        print(f"  [SUFFICIENCY CHECK] Error: {e}")
        return {"is_sufficient": False, "missing_facts": [f"Error: {e}"], "reason": f"Check error: {e}"}


# ---------------------------------------------------------------------------
# search_advisories — returns raw chunks including chunk_text for CE gate
# ---------------------------------------------------------------------------

def search_advisories(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Runs FAISS retrieval; returns ALL top_k chunks including chunk_text for CE gate."""
    retriever = get_retriever()
    # Retrieve without score floor — gate is applied in run_agent deterministically
    results = retriever.search(query=query, top_k=top_k, min_score=0.0)
    cleaned = []
    for r in results:
        cleaned.append({
            "doc_name": r.get("doc_name", "Unknown Document"),
            "topic": r.get("topic", "general"),
            "section": r.get("section", "General"),
            "snippet": r.get("chunk_text", "")[:350] + "...",
            "chunk_text": r.get("chunk_text", ""),   # kept for CE gate
            "score": r.get("score", 0.0),
            "source_type": "knowledge_base"
        })
    return cleaned


# ---------------------------------------------------------------------------
# URL allowlist and page date extraction
# ---------------------------------------------------------------------------

import urllib.parse

ALLOWED_HOSTS = [
    "who.int", "mohfw.gov.in", "nhp.gov.in", "icmr.gov.in",
    "ncvbdc.mohfw.gov.in", "tbcindia.mohfw.gov.in", "cdc.gov", "ncbi.nlm.nih.gov",
    "archive.cdc.gov"
]


def is_url_allowed(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower().split(":")[0]
        allowed = any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)
        if not allowed:
            return False
        if "ncbi.nlm.nih.gov" in host and not parsed.path.startswith("/books/"):
            return False
        return True
    except Exception:
        return False


def extract_page_date(soup) -> Optional[str]:
    """
    Extracts page date from meta tags in priority order:
    DC.date, DC.Date, DC.date.issued, article:modified_time,
    article:published_time, cdc:last_updated, last-modified,
    date, then falls back to <time> tag, then on-page text scan
    (WHO prints dates under the H1, CDC in specific spans).
    """
    try:
        META_NAMES = [
            "DC.date", "DC.Date", "dc.date", "DC.date.issued",
            "cdc:last_updated", "last-modified", "date",
        ]
        META_PROPS = [
            "article:modified_time", "article:published_time",
        ]
        for meta_name in META_NAMES:
            meta = soup.find("meta", attrs={"name": re.compile(f"^{re.escape(meta_name)}$", re.I)})
            if meta and meta.get("content"):
                return meta.get("content").strip()

        for prop in META_PROPS:
            meta = soup.find("meta", attrs={"property": re.compile(f"^{re.escape(prop)}$", re.I)})
            if meta and meta.get("content"):
                return meta.get("content").strip()

        # On-page <time> fallback
        time_tag = soup.find("time")
        if time_tag:
            dt = time_tag.get("datetime") or time_tag.get_text(strip=True)
            if dt:
                return dt

        # On-page text scan: WHO prints date near H1, CDC in spans
        DATE_PATTERNS = [
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
            r'(\d{4}[-/]\d{2}[-/]\d{2})',
        ]
        # Check near H1 first
        h1 = soup.find("h1")
        if h1:
            for sib in h1.find_all_next(limit=5):
                sib_text = sib.get_text(strip=True)
                for pat in DATE_PATTERNS:
                    m = re.search(pat, sib_text, re.I)
                    if m:
                        return m.group(1)
        # Broader scan of page header area
        header_area = soup.find("header") or soup.find(class_=re.compile(r"date|published|updated", re.I))
        if header_area:
            header_text = header_area.get_text(strip=True)
            for pat in DATE_PATTERNS:
                m = re.search(pat, header_text, re.I)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return None


def parse_html_tables(soup) -> List[Dict[str, Any]]:
    """
    Column-aware HTML table parser. Extracts tables into list of dicts,
    where each dict maps column headers to cell values.
    Uses get_text(separator=" ") to prevent cell text squashing across newlines.
    """
    parsed_tables = []
    try:
        tables = soup.find_all("table")
        for tbl_idx, table in enumerate(tables[:5]):  # cap at 5 tables
            rows = table.find_all("tr")
            if not rows:
                continue

            # Extract headers from first row using separator=" " to avoid squash
            header_cells = rows[0].find_all(["th", "td"])
            headers = [cell.get_text(separator=" ", strip=True) for cell in header_cells]
            if not headers or all(h == "" for h in headers):
                continue

            table_data = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                row_data = {}
                for col_idx, cell in enumerate(cells):
                    header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                    row_data[header] = cell.get_text(separator=" ", strip=True)
                if row_data:
                    table_data.append(row_data)

            if table_data:
                parsed_tables.append({
                    "table_index": tbl_idx,
                    "headers": headers,
                    "rows": table_data
                })
    except Exception as e:
        print(f"  [TABLE PARSE] Error: {e}")
    return parsed_tables


TOOL_CALLING_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
NON_TOOL_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

def _get_msg_str(m: Any) -> str:
    if isinstance(m, dict):
        return str(m.get("content") or "")
    if hasattr(m, "content"):
        return str(getattr(m, "content") or "")
    return ""

def safe_chat_completion(client: Groq, active_model: str, **kwargs):
    """
    Executes a Groq chat completion.
    Respects circuit-breaker per-model cooldowns before trying.
    If 429, records Retry-After header and marks model cooling.
    Also prunes/truncates oversized messages if total payload exceeds ~16,000 characters.
    When every model is cooling: fails fast with a clear busy message.
    """
    # Prevent token limit (TPM 8k) overflow by truncating long message contents
    if "messages" in kwargs and kwargs["messages"]:
        msgs = kwargs["messages"]
        total_chars_before = sum(len(_get_msg_str(m)) for m in msgs)
        if total_chars_before > 16000:
            new_msgs = []
            truncated_count = 0
            for m in msgs:
                if isinstance(m, dict):
                    m_copy = dict(m)
                    content = m_copy.get("content")
                    if isinstance(content, str) and len(content) > 1500 and m_copy.get("role") in ["tool", "user"]:
                        m_copy["content"] = content[:1500] + "\n...[truncated for token limit]..."
                        truncated_count += 1
                    new_msgs.append(m_copy)
                else:
                    new_msgs.append(m)
            kwargs["messages"] = new_msgs
            total_chars_after = sum(len(_get_msg_str(m)) for m in kwargs["messages"])
            print(f"  [CONTEXT PRUNING] Payload chars before={total_chars_before} -> after={total_chars_after} ({truncated_count} messages truncated)")

    is_tool_call = bool(kwargs.get("tools"))
    candidate_list = TOOL_CALLING_MODELS if is_tool_call else NON_TOOL_MODELS
    models_to_try = [active_model] + [m for m in candidate_list if m != active_model]

    # Filter out cooling models
    available = _available_models_in_order(models_to_try)
    if not available:
        cooling_info = {m: round(_model_cooldown_until.get(m, 0) - time.time(), 0) for m in models_to_try}
        raise RuntimeError(
            f"All models are cooling down (circuit breaker active). "
            f"Cooldown remaining (s): {cooling_info}. Please try again later."
        )

    last_err = None
    for m in available:
        try:
            kwargs["model"] = m
            res = client.chat.completions.create(**kwargs)
            served_model = getattr(res, "model", m)
            print(f"  [GROQ CALL SUCCESS] Target model: '{m}' | Served model: '{served_model}'")
            return res
        except Exception as e:
            last_err = e
            err_str = str(e)
            # Parse Retry-After if present (Groq may include it in error body)
            retry_after = None
            try:
                # Check for retry_after in headers if accessible
                if hasattr(e, "response") and e.response is not None:
                    ra = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                    if ra:
                        retry_after = int(ra)
            except Exception:
                pass

            if any(k in err_str.lower() for k in ["429", "413", "rate_limit", "limit", "too large", "not supported", "tpm"]):
                _mark_model_429(m, retry_after_secs=retry_after)
                print(f"  [MODEL FALLBACK] Model '{m}' error ({err_str[:70]}). Trying next available model...")
                continue
            else:
                raise e
    raise last_err


# ---------------------------------------------------------------------------
# web_search_health_authority
# ---------------------------------------------------------------------------

def extract_passages_and_rerank(query: str, raw_text: str, top_k_passages: int = 3) -> str:
    """Splits text into ~800-char passages and reranks using CrossEncoder."""
    if not raw_text.strip():
        return ""

    paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 30]
    passages = []
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) < 800:
            buffer += ("\n\n" if buffer else "") + p
        else:
            if buffer:
                passages.append(buffer)
            buffer = p
    if buffer:
        passages.append(buffer)

    if not passages:
        return raw_text[:1500]

    ce = _get_cross_encoder()
    if not ce:
        return "\n\n---\n\n".join(passages[:top_k_passages])

    try:
        pairs = [(query, p[:512]) for p in passages]
        scores = ce.predict(pairs)
        ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        top_passages = [p for _, p in ranked[:top_k_passages]]
        return "\n\n---\n\n".join(top_passages)
    except Exception:
        return "\n\n---\n\n".join(passages[:top_k_passages])


def _extract_pdf_columns(page) -> str:
    """
    Column-aware PDF text extraction using pdfplumber.
    Clusters words into left and right columns based on x-midline,
    then reads each column top-to-bottom before joining.
    Only splits when the page is a true two-column layout (no full-width text crosses midline).
    A true spread: width > height * 1.2 AND no word straddles the midline (x0 < mid AND x1 > mid).
    """
    w = page.width
    h = page.height
    mid = w / 2.0

    if w <= h * 1.2:
        # Portrait page: single column extraction
        return page.extract_text(layout=False) or ""

    # Landscape: check if it's a true two-column spread or a single-column landscape
    words = page.extract_words()
    if not words:
        return page.extract_text(layout=False) or ""

    # Count words that straddle the midline
    straddle_count = sum(1 for wrd in words if wrd["x0"] < mid and wrd["x1"] > mid)
    total_words = len(words)
    straddle_fraction = straddle_count / total_words if total_words > 0 else 0

    # Zero-straddle rule: any word straddling midline -> full-width text, NOT a true spread
    if straddle_count > 0:
        return page.extract_text(layout=False) or ""

    # True two-column spread: cluster by midline, sort each column top-to-bottom by y0
    left_words = sorted([w for w in words if w["x0"] < mid], key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))
    right_words = sorted([w for w in words if w["x0"] >= mid], key=lambda w: (round(w["top"] / 5) * 5, w["x0"]))

    def words_to_text(word_list):
        if not word_list:
            return ""
        lines = []
        current_line = []
        current_top = None
        for wrd in word_list:
            top = round(wrd["top"] / 5) * 5
            if current_top is None or abs(top - current_top) <= 5:
                current_line.append(wrd["text"])
                current_top = top
            else:
                lines.append(" ".join(current_line))
                current_line = [wrd["text"]]
                current_top = top
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(lines)

    left_text = words_to_text(left_words)
    right_text = words_to_text(right_words)
    return (left_text + "\n\n" + right_text).strip()


def extract_pdf_content(content_bytes: bytes, filename: str) -> tuple:
    """
    Extracts text and title from PDF content bytes via pdfplumber or fitz.
    Uses column-aware extraction for true two-column spreads.
    A true spread: landscape (width > height * 1.2) AND zero words straddle midline.
    Uses first heading/line as title if doc title is 'Untitled-1' or default.
    """
    clean_title = filename.replace(".pdf", "").replace("_", " ").title()
    text = ""
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            page_texts = []
            for p in pdf.pages:
                p_txt = _extract_pdf_columns(p)
                if p_txt.strip():
                    page_texts.append(p_txt.strip())
            text = "\n\n".join(page_texts)

            if pdf.metadata and pdf.metadata.get("Title"):
                t = pdf.metadata.get("Title").strip()
                if len(t) > 3 and not t.lower().endswith(".pdf") and t.lower() != "untitled-1":
                    clean_title = t

            if clean_title.lower() == "untitled-1" or clean_title == filename.replace(".pdf", "").replace("_", " ").title():
                lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 5]
                if lines:
                    clean_title = lines[0][:80]
    except Exception as e:
        print(f"    [PDF EXTRACTION] pdfplumber error: {e}")
        try:
            import fitz, io
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            page_texts = []
            for page in doc:
                rect = page.rect
                if rect.width > rect.height * 1.2:
                    left_rect = fitz.Rect(0, 0, rect.width / 2, rect.height)
                    right_rect = fitz.Rect(rect.width / 2, 0, rect.width, rect.height)
                    p_txt = (page.get_text(clip=left_rect) or "") + "\n\n" + (page.get_text(clip=right_rect) or "")
                else:
                    p_txt = page.get_text() or ""
                if p_txt.strip():
                    page_texts.append(p_txt.strip())
            text = "\n\n".join(page_texts)
        except Exception as e2:
            print(f"    [PDF EXTRACTION] fitz error: {e2}")

    return text, clean_title


def web_search_health_authority(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Live domain-restricted web search fallback with strict URL allowlist filter, passage reranking, and date extraction."""
    raw_results = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddg = DDGS()
        search_query = (
            f"{query} site:who.int OR site:mohfw.gov.in OR site:nhp.gov.in "
            f"OR site:icmr.gov.in OR site:ncvbdc.mohfw.gov.in OR site:tbcindia.mohfw.gov.in "
            f"OR site:cdc.gov OR site:ncbi.nlm.nih.gov"
        )
        raw_results = list(ddg.text(search_query, max_results=max_results * 3))

        if not raw_results:
            fallback_query = f"{query} site:who.int OR site:cdc.gov OR site:mohfw.gov.in"
            raw_results = list(ddg.text(fallback_query, max_results=max_results * 2))
    except Exception as e:
        print(f"Web search error: {e}")
        raw_results = []

    if not raw_results:
        return []

    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Upgrade-Insecure-Requests": "1"
    }

    results = []
    for item in raw_results:
        if len(results) >= max_results:
            break

        url = item.get("href") or item.get("link") or ""
        title = item.get("title") or "Public Health Guidance"
        snippet = item.get("body") or item.get("snippet") or ""

        if not url or not is_url_allowed(url):
            print(f"  [ALLOWLIST FILTER] Dropped disallowed or non-books URL: {url}")
            continue

        page_text = ""
        page_date = None
        http_status = None
        content_type = None
        bytes_len = 0
        content_source = "search_snippet"

        try:
            resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            http_status = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
            bytes_len = len(resp.content)

            print(f"  [WEB FETCH LOG] URL: {url} | Status: {http_status} | Content-Type: {content_type} | Bytes: {bytes_len}")

            if resp.status_code == 200:
                if "application/pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                    filename = url.split("/")[-1]
                    pdf_text, pdf_title = extract_pdf_content(resp.content, filename)
                    if pdf_text:
                        page_text = pdf_text
                        title = pdf_title
                        content_source = "page_fetch"
                else:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    page_date = extract_page_date(soup)

                    # Column-aware table parsing
                    parsed_tables = parse_html_tables(soup)

                    for element in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
                        element.decompose()
                    main_elem = soup.find("main") or soup.find("article") or soup.find("body")
                    if main_elem:
                        text_content = main_elem.get_text(separator=" ", strip=True)
                        text_content = re.sub(r"\s+", " ", text_content)
                        if parsed_tables:
                            tbl_str = "\n\nPARSED TABLES:\n" + repr(parsed_tables[:2])[:600]
                            text_content += tbl_str
                        if len(text_content) > 100:
                            page_text = text_content
                            content_source = "page_fetch"
        except Exception as e:
            print(f"  [WEB FETCH LOG] Failed to fetch {url}: {e}")

        # Passage splitting and reranking
        if page_text:
            reranked_snippet = extract_passages_and_rerank(query, page_text, top_k_passages=3)
            final_snippet = reranked_snippet[:1000] if reranked_snippet else page_text[:1000]
        else:
            final_snippet = snippet[:1000]
            content_source = "search_snippet"

        print(f"  [FETCH REPR (First 300 chars)]: {repr(final_snippet[:300])}")

        results.append({
            "doc_name": title,
            "title": title,
            "source_url": url,
            "page_date": page_date,
            "http_status": http_status,
            "content_type": content_type,
            "bytes_len": bytes_len,
            "content_source": content_source,
            "section": "Live Web Search Result",
            "snippet": final_snippet,
            "content_snippet": final_snippet,
            "score": 0.90,
            "source_type": "live_search"
        })

    return results


# ---------------------------------------------------------------------------
# check_myth
# ---------------------------------------------------------------------------

def check_myth(query: str) -> Dict[str, Any]:
    """Checks query against curated seed database of medical myths."""
    myths_path = os.path.join(BACKEND_DIR, "data", "myths.json")
    if not os.path.exists(myths_path):
        myths_path = os.path.join(os.path.dirname(BACKEND_DIR), "data", "myths.json")

    if not os.path.exists(myths_path):
        return {"found_myth": False, "message": "Myth database file missing."}

    with open(myths_path, "r", encoding="utf-8") as f:
        myths = json.load(f)

    q_lower = query.lower()
    for item in myths:
        myth_str = item["myth"].lower()
        myth_words = [w for w in myth_str.split() if len(w) > 3]
        overlap = sum(1 for w in myth_words if w in q_lower)

        if myth_str in q_lower or q_lower in myth_str or overlap >= 2:
            return {
                "found_myth": True,
                "myth": item["myth"],
                "fact": item["fact"],
                "topic": item.get("topic", "general")
            }

    return {"found_myth": False, "message": "No matching health myth found for this query."}


# ---------------------------------------------------------------------------
# get_session_history
# ---------------------------------------------------------------------------

def get_session_history(user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Queries symptom_logs table in Neon/PostgreSQL for user history using connection pool."""
    db = SessionLocal()
    try:
        logs = db.query(SymptomLog).filter(SymptomLog.user_id == user_id).order_by(SymptomLog.created_at.desc()).limit(limit).all()
        history = []
        for l in logs:
            history.append({
                "id": l.id,
                "user_id": l.user_id,
                "query_text": l.query_text,
                "topic": l.topic,
                "triage_tag": l.triage_tag,
                "created_at": l.created_at.isoformat() if l.created_at else None
            })
        return history
    except Exception as e:
        return [{"error": f"Failed to fetch session history: {str(e)}"}]
    finally:
        db.close()


def _async_db_log(user_id: str, query_text: str, topic: str, triage_tag: str):
    """Background worker for non-blocking DB symptom logging."""
    db = SessionLocal()
    try:
        log_entry = SymptomLog(
            user_id=user_id,
            query_text=query_text,
            topic=topic or "general",
            triage_tag=triage_tag or "GENERAL_INFO"
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Async DB log error: {e}")
    finally:
        db.close()


def log_symptom(user_id: str, query_text: str, topic: str, triage_tag: str) -> Dict[str, Any]:
    """Schedules background insertion into symptom_logs table for zero blocking latency."""
    bg_executor.submit(_async_db_log, user_id, query_text, topic, triage_tag)
    return {
        "status": "success",
        "message": "Symptom log scheduled asynchronously",
        "user_id": user_id,
        "topic": topic,
        "triage_tag": triage_tag
    }


def triage_classify(query: str, retrieved_context: str) -> Dict[str, Any]:
    """Fast classification of medical urgency into EMERGENCY, CONSULT_SOON, SELF_CARE, or GENERAL_INFO."""
    q_low = query.lower()

    # Fast heuristic check for emergency red flags
    if "chest pain" in q_low or "can't breathe" in q_low or "difficulty breathing" in q_low or "severe bleeding" in q_low:
        return {"triage_tag": "EMERGENCY", "confidence": 95, "reason": "Red flag symptoms indicating acute life threat."}

    # Fast heuristic check for myth / info queries
    if "myth" in q_low or "turmeric" in q_low or "cure" in q_low:
        return {"triage_tag": "GENERAL_INFO", "confidence": 90, "reason": "General health query or myth check."}

    # Lightweight LLM call if non-red flag
    client = Groq(api_key=settings.GROQ_API_KEY)
    prompt = (
        f'Classify urgency for: "{query}". Context: "{retrieved_context[:200]}".\n'
        'Options: EMERGENCY, CONSULT_SOON, SELF_CARE, GENERAL_INFO.\n'
        'JSON response: {"triage_tag": "<TAG>", "confidence": 85, "reason": "<reason>"}'
    )

    try:
        res = client.chat.completions.create(
            model=settings.GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return {
            "triage_tag": data.get("triage_tag", "CONSULT_SOON"),
            "confidence": data.get("confidence", 85),
            "reason": data.get("reason", "Evaluated clinical urgency.")
        }
    except Exception:
        return {"triage_tag": "CONSULT_SOON", "confidence": 75, "reason": "Symptom evaluation recommended."}


def get_regional_alerts(state: str = "Punjab") -> Dict[str, Any]:
    """Placeholder stub for regional IDSP outbreak surveillance data."""
    return {
        "state": state,
        "status": "No active disease outbreak alerts registered.",
        "source": "IDSP Surveillance Feed (Stub)",
        "note": "TODO: Wire up live IDSP outbreak data in future iteration."
    }


# ---------------------------------------------------------------------------
# Tool registry and Groq schema
# ---------------------------------------------------------------------------

TOOLS_MAP = {
    "search_advisories": search_advisories,
    "web_search_health_authority": web_search_health_authority,
    "check_myth": check_myth,
    "get_session_history": get_session_history,
    "log_symptom": log_symptom,
    "triage_classify": triage_classify,
    "get_regional_alerts": get_regional_alerts
}

GROQ_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_advisories",
            "description": "Searches public health guidelines (29 topics: dengue, TB, cardiac, diabetes, malaria, HIV, etc.) for verified clinical guidance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Health or medical topic query to search"},
                    "top_k": {"type": "integer", "description": "Number of top matching chunks to retrieve (default 5)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_health_authority",
            "description": "Performs live web search restricted to trusted public health authority domains (WHO, MoHFW, NHP, CDC, ICMR) as a fallback when search_advisories has no direct match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The health query or topic to search on official health portals"},
                    "max_results": {"type": "integer", "description": "Max search results to return (default 3)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_myth",
            "description": "Checks if a health question or claim matches a known medical myth or misinformation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The health claim or myth to check"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_history",
            "description": "Fetches past symptom logs and previous user interactions for context continuity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier"},
                    "limit": {"type": "integer", "description": "Max history items to return (default 5)"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_symptom",
            "description": "Logs the current interaction, topic, and triage level into the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier"},
                    "query_text": {"type": "string", "description": "Original query text"},
                    "topic": {"type": "string", "description": "Topic e.g. dengue, tuberculosis, maternal_child_health, general"},
                    "triage_tag": {"type": "string", "description": "Triage classification (EMERGENCY, CONSULT_SOON, SELF_CARE, GENERAL_INFO)"}
                },
                "required": ["user_id", "query_text", "topic", "triage_tag"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "triage_classify",
            "description": "Evaluates medical urgency into EMERGENCY, CONSULT_SOON, SELF_CARE, or GENERAL_INFO.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user query text"},
                    "retrieved_context": {"type": "string", "description": "Retrieved health guideline context or myth info"}
                },
                "required": ["query", "retrieved_context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_regional_alerts",
            "description": "Retrieves regional disease outbreak alerts for a specific Indian state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "State name e.g. Punjab, Delhi"}
                },
                "required": []
            }
        }
    }
]


def _execute_single_tool(tc: Any) -> Dict[str, Any]:
    fn_name = tc.function.name
    t0 = time.time()
    try:
        fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
    except Exception:
        fn_args = {}

    if fn_name in TOOLS_MAP:
        tool_fn = TOOLS_MAP[fn_name]
        try:
            tool_result = tool_fn(**fn_args)
        except Exception as e:
            tool_result = {"error": f"Tool execution failed: {str(e)}"}
    else:
        tool_result = {"error": f"Unknown tool name '{fn_name}'"}

    dt = round((time.time() - t0) * 1000, 2)
    return {
        "tc_id": tc.id,
        "fn_name": fn_name,
        "args": fn_args,
        "result": tool_result,
        "duration_ms": dt
    }


# ---------------------------------------------------------------------------
# Grounding Audit — with auditor model different from generator
# ---------------------------------------------------------------------------

def _pick_auditor_model(generator_served_model: str, candidate_chain: List[str]) -> Optional[str]:
    """
    Choose an auditor model that is DIFFERENT from the generator's served model.
    Excludes cooling models via circuit breaker.
    Returns None if no different model is available.
    """
    for m in candidate_chain:
        if m != generator_served_model and not _model_is_cooling(m):
            return m
    return None


def verify_grounding(client: Groq, active_model: str, answer_to_audit: str,
                     sources: List[Dict[str, Any]],
                     generator_served_model: str = None) -> Dict[str, Any]:
    """
    Audits whether every sentence in answer_to_audit is supported by reference sources.
    Uses a model DIFFERENT from the generator's served model.
    If no different model is available, marks answer as unverified.
    Prints model match warning if generator and auditor served model match.
    """
    audit_chain = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    served = generator_served_model or active_model
    auditor_model = _pick_auditor_model(served, audit_chain)

    if auditor_model is None:
        print(f"  [AUDIT] No different model available (all cooling or only {served} available). Marking as unverified.")
        return {
            "unsupported_sentences": [],
            "is_fully_grounded": None,
            "audit_summary": f"Unverified: no auditor model different from generator ({served}) was available.",
            "audit_messages": [],
            "auditor_model": None
        }

    if auditor_model == served:
        print(f"  [WARNING: MODEL MATCH] Generator and auditor are the same model: '{served}'. Self-auditing is unreliable.")

    if not sources:
        ref_text = "No reference sources available (ungrounded general knowledge)."
    else:
        source_parts = []
        for s in sources:
            name = s.get("doc_name") or s.get("title") or "Unknown"
            stype = s.get("source_type", "")
            snip = (s.get("snippet") or s.get("content_snippet") or "")[:400]
            source_parts.append(f"- Source ({stype}): {name}\n  Snippet: {snip}")
        ref_text = "\n\n".join(source_parts)[:2500]

    answer_truncated = answer_to_audit[:1500]

    audit_prompt = (
        "You are a strict clinical auditor evaluating AI medical grounding.\n\n"
        f"Reference Sources:\n{ref_text}\n\n"
        f"Generated Answer (to audit):\n\"{answer_truncated}\"\n\n"
        "Task: For each sentence in the Generated Answer, identify any factual clinical claim NOT "
        "supported by the Reference Sources above.\n"
        "If all factual claims are supported, set unsupported_sentences to [].\n\n"
        "EXCEPTION RULES FOR AUDITOR:\n"
        "- Do NOT flag standard clinical caveats ('Because these infections share similar symptoms, they are hard to tell apart, and laboratory testing is needed for a clear diagnosis') "
        "as unsupported_sentences. These are required safety elements.\n"
        "- Do NOT flag standard doctor disclaimers ('Of course, I'm not a substitute for your doctor') as unsupported.\n"
        "- Do NOT flag the standard treatment closing ('a doctor chooses the treatment, especially in pregnancy or for young children') as unsupported.\n\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"unsupported_sentences": ["<sentence>", "..."], "is_fully_grounded": true/false, '
        '"audit_summary": "<one sentence>"}'
    )

    for attempt in range(1, 3):  # try at most twice
        res = None
        try:
            res = safe_chat_completion(
                client,
                auditor_model,
                messages=[{"role": "user", "content": audit_prompt}],
                temperature=0.0,
                max_tokens=1500
            )
            served_model = getattr(res, "model", auditor_model)
            raw = res.choices[0].message.content or ""
            finish_reason = getattr(res.choices[0], "finish_reason", "completed")

            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = re.sub(r"^```[a-zA-Z]*\n?", "", raw_clean)
                raw_clean = re.sub(r"\n?```$", "", raw_clean).strip()
            match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
            if match:
                raw_clean = match.group(0)

            data = json.loads(raw_clean)
            if "is_fully_grounded" not in data:
                raise ValueError("Missing is_fully_grounded key in JSON response")

            print(f"  [GROUNDING AUDIT SUCCESS] Model: '{served_model}' | Finish Reason: '{finish_reason}' | Grounded: {data.get('is_fully_grounded')}")
            if served_model == served:
                print(f"  [WARNING: MODEL MATCH] Auditor served model '{served_model}' matches generator served model '{served}'.")

            return {
                "unsupported_sentences": data.get("unsupported_sentences", []),
                "is_fully_grounded": data.get("is_fully_grounded"),
                "audit_summary": data.get("audit_summary", "Audit complete."),
                "audit_messages": [{"role": "user", "content": audit_prompt}, {"role": "assistant", "content": raw}],
                "auditor_model": served_model
            }
        except Exception as e:
            raw_content = ""
            finish_reason = "N/A"
            served_model = auditor_model
            if res and res.choices:
                raw_content = res.choices[0].message.content or ""
                finish_reason = getattr(res.choices[0], "finish_reason", "unknown")
                served_model = getattr(res, "model", auditor_model)

            print(f"  [AUDIT FAILED] Attempt {attempt} | Model: '{served_model}' | Finish Reason: '{finish_reason}'")
            print(f"  [AUDIT RAW CONTENT]: {repr(raw_content[:300])}")
            print(f"  [AUDIT ERROR]: {e}")
            if attempt == 2:
                print("  [AUDIT] Both attempts failed — setting is_fully_grounded = null (unverified)")
                return {
                    "unsupported_sentences": [],
                    "is_fully_grounded": None,
                    "audit_summary": f"Audit unverified after 2 attempts: {str(e)[:200]}",
                    "audit_messages": [{"role": "user", "content": audit_prompt}],
                    "auditor_model": None
                }
            time.sleep(0.5)

    return {"unsupported_sentences": [], "is_fully_grounded": None, "audit_summary": "Audit failed.", "auditor_model": None}


def _regenerate_constrained(client: Groq, active_model: str, query: str,
                             original_answer: str, sources: List[Dict[str, Any]],
                             unsupported_sentences: List[str]) -> str:
    """
    When audit returns unsupported sentences, regenerate the answer constrained
    strictly to the provided sources, removing any extrapolation.
    Extractive framing: paraphrase only what is explicitly in the passages.
    No markdown tables or headers (use plain text and bullet lists only for 3+ items).
    """
    if not unsupported_sentences:
        return original_answer

    source_parts = []
    for s in sources:
        name = s.get("doc_name") or s.get("title") or "Unknown"
        snip = (s.get("snippet") or s.get("content_snippet") or "")[:500]
        source_parts.append(f"- {name}: {snip}")
    source_text = "\n".join(source_parts)[:2000]

    regen_prompt = (
        f"You are SwasthyaSaathi. The following answer to \"{query}\" was found to contain "
        f"sentences not supported by the reference sources:\n\n"
        f"Unsupported sentences to REMOVE:\n"
        + "\n".join(f"  - {s}" for s in unsupported_sentences)
        + f"\n\nOriginal answer:\n\"{original_answer[:1200]}\"\n\n"
        f"Reference sources (ONLY use facts from these):\n{source_text}\n\n"
        "Rewrite the answer in the same conversational tone, removing or replacing all unsupported "
        "claims. ONLY include information explicitly present in the reference sources. "
        "IMPORTANT FORMATTING RULES:\n"
        "- Do NOT use markdown tables (no | --- | columns).\n"
        "- Do NOT use markdown headers (no #, ##, ###).\n"
        "- Use plain sentences and bullet lists only for 3 or more distinct items.\n"
        "For look-alike diseases (such as Chikungunya vs Dengue), ALWAYS preserve the sentence: "
        "'Because these infections share similar symptoms, they are hard to tell apart, and laboratory testing is needed for a clear diagnosis.' "
        "End with the standard 'Of course, I'm not a substitute for your doctor' closing."
    )

    try:
        res = safe_chat_completion(
            client,
            active_model,
            messages=[{"role": "user", "content": regen_prompt}],
            temperature=0.1,
            max_tokens=700,
        )
        regen = res.choices[0].message.content
        if regen and len(regen) > 50:
            print(f"  [REGEN] Answer regenerated — removed {len(unsupported_sentences)} unsupported sentence(s).")
            return regen
        return original_answer
    except Exception as e:
        print(f"  [REGEN] Regeneration failed: {e}. Keeping original.")
        return original_answer


# ---------------------------------------------------------------------------
# Comparison query detection
# ---------------------------------------------------------------------------
# A comparison query contains explicit contrast language referencing 2+ diseases.
_COMPARISON_KEYWORDS = [
    r"\bvs\b", r"\bversus\b", r"\bdifferent from\b", r"\bdifference between\b",
    r"\bcompare\b", r"\bdistinguish\b", r"\btell apart\b", r"\bsimilar to\b",
    r"\bsame as\b", r"\bor\b.{1,30}\b(fever|infection|disease|virus|bacteria)\b"
]

def is_comparison_query(query: str) -> bool:
    """Returns True if query explicitly compares two or more diseases/conditions."""
    q = query.lower()
    for pattern in _COMPARISON_KEYWORDS:
        if re.search(pattern, q):
            return True
    return False


# ---------------------------------------------------------------------------
# Model verification
# ---------------------------------------------------------------------------

def _verify_model_available(client: Groq, model_name: str) -> bool:
    """Prints available models from client.models.list() and checks if model_name is present."""
    try:
        available = [m.id for m in client.models.list().data]
        print(f"  [MODEL CHECK] Active model string in API call: '{model_name}'")
        print(f"  [MODEL CHECK] Available models via client.models.list(): {available}")
        if model_name not in available:
            print(f"  [MODEL CHECK] WARNING: '{model_name}' not in listed models. Continuing anyway.")
            return False
        return True
    except Exception as e:
        print(f"  [MODEL CHECK] models.list() failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Orchestrator Core Loop
# ---------------------------------------------------------------------------

def run_agent(user_id: str, query: str) -> Dict[str, Any]:
    """
    Agentic Orchestrator with:
    - CE < 0: skip KB and sufficiency check entirely, go straight to web search
    - CE >= 0: run sufficiency check in parallel with live web search
    - Draft → audit(draft) → [regen if needed] → final → audit(final)
    - Auditor model chosen to differ from generator served model
    - Doctor note appended after audit (not filterable by regen)
    - Testing caveat only for comparison queries
    - Latency targets: <10s if draft passes audit, <20s with regen
    """
    init_db()
    overall_start = time.time()
    active_model = settings.GROQ_MODEL_NAME
    client = Groq(api_key=settings.GROQ_API_KEY)

    is_comparison = is_comparison_query(query)
    print(f"\n[ORCHESTRATOR START] Model: '{active_model}' | User: '{user_id}' | Query: '{query}'")
    print(f"  [QUERY TYPE] comparison={is_comparison}")
    _verify_model_available(client, active_model)

    timing_breakdown = []
    sources_collected = []
    fallback_fired = False

    # 1. RETRIEVAL & SPECIFICITY GATE
    t_ret_start = time.time()
    raw_chunks = search_advisories(query, top_k=5)
    dt_ret = round((time.time() - t_ret_start) * 1000, 2)
    timing_breakdown.append({"step": "Parallel Tools Execution (search_advisories)", "duration_ms": dt_ret})

    t_gate_start = time.time()
    gate_result = specificity_gate(query, raw_chunks, faiss_floor=0.45, encode_query_str=query)
    dt_gate = round((time.time() - t_gate_start) * 1000, 2)
    timing_breakdown.append({"step": "CE Specificity Gate", "duration_ms": dt_gate})

    max_ce = gate_result.get("max_ce_score")
    skip_kb = gate_result.get("skip_kb", False)

    # 2. PARALLEL SUFFICIENCY CHECK & LIVE SEARCH (or hard skip)
    suff_result = None
    triage_tag = "GENERAL_INFO"
    missing_facts = []

    if skip_kb or max_ce is None:
        # CE < 0: hard skip — no sufficiency check, go straight to web search
        print(f"  [GATE] Max-CE={max_ce} < 0 → hard skip KB and sufficiency check.")
        fallback_fired = True
        t_fb_start = time.time()
        live_res = web_search_health_authority(query=query, max_results=3)
        dt_fb = round((time.time() - t_fb_start) * 1000, 2)
        timing_breakdown.append({"step": "Live Web Search & Rerank", "duration_ms": dt_fb})
        for s in live_res:
            s["source_type"] = "live_search"
        sources_collected.extend(live_res)
    else:
        # CE >= 0: run sufficiency check and live web search in parallel
        t_parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_suff = executor.submit(
                sufficiency_check, client, active_model, query, gate_result["valid_chunks"]
            )
            fut_triage = executor.submit(
                triage_classify, query,
                "\n".join([c.get("snippet", "") for c in gate_result["valid_chunks"]])
            )
            fut_web = executor.submit(web_search_health_authority, query, 3)

            suff_result = fut_suff.result()
            triage_res = fut_triage.result()
            live_res_parallel = fut_web.result()

        dt_parallel = round((time.time() - t_parallel_start) * 1000, 2)
        timing_breakdown.append({"step": "Parallel (Sufficiency + Live Search + Triage)", "duration_ms": dt_parallel})

        if isinstance(triage_res, dict) and "triage_tag" in triage_res:
            triage_tag = triage_res["triage_tag"]

        if suff_result.get("is_sufficient", False):
            # KB is sufficient — use KB sources, no web fallback
            sources_collected.extend(gate_result["valid_chunks"])
            print(f"  [SUFFICIENCY] KB is sufficient. Using KB chunks only.")
        else:
            # KB not sufficient — use web results
            missing_facts = suff_result.get("missing_facts", [])
            gate_result["reason"] += f" | Sufficiency check failed: {suff_result.get('reason')} (missing: {missing_facts})"
            fallback_fired = True
            for s in live_res_parallel:
                s["source_type"] = "live_search"
            sources_collected.extend(live_res_parallel)
            print(f"  [SUFFICIENCY] KB insufficient. Using web fallback.")

    # 3. SINGLE GENERATION CALL
    system_prompt = (
        "You are SwasthyaSaathi — a warm, knowledgeable public health companion grounded in official guidelines "
        "from MoHFW, ICMR, NHP, and WHO.\n\n"

        "TONE AND FORMAT RULES:\n"
        "- Write in flowing, natural sentences like a caring, knowledgeable person would speak.\n"
        "- Only use bullet points when listing 3 or more distinct steps or items.\n"
        "- Do NOT use markdown tables (no | --- | columns).\n"
        "- Do NOT use markdown headers (no #, ##, ###).\n"
        "- Mention sources naturally within a sentence ('According to WHO guidelines...', 'The MoHFW advises...').\n\n"

        "EXTRACTIVE PARAPHRASING RULE (STRICT):\n"
        "- Paraphrase ONLY what is explicitly present in the provided reference passages.\n"
        "- Do NOT add clinical details, drug names, or facts not in the provided sources.\n\n"

        "MEDICATION SAFETY RULE (STRICT MANDATORY):\n"
        "- NEVER include specific milligram dosages (e.g., '100 mg', '500 mg', '30-50 mg/kg'), dosing frequencies (e.g., 'twice daily', 'every 6 hours'), or treatment durations (e.g., 'for 7 days').\n"
        "- You may mention recommended drug names (e.g., Doxycycline, Ampicillin), but ALL dosages, schedules, and durations MUST be stated as 'as prescribed by a doctor'.\n\n"

        "SELF-CONTAINMENT RULE (STRICT):\n"
        "- NEVER reference previous answers, earlier context, or prior turns ('as I mentioned', 'what I shared above').\n"
        "- Each answer must be fully self-contained.\n\n"

        "LOOK-ALIKE DISEASES RULE (STRICT MANDATORY — COMPARISON QUERIES ONLY):\n"
        "- Whenever answering questions comparing similar/look-alike diseases (such as Chikungunya vs Dengue), "
        "you MUST include this exact sentence verbatim: "
        "'Because these infections share similar symptoms, they are hard to tell apart, and laboratory testing is needed for a clear diagnosis.'\n"
        "- Do NOT include this sentence for non-comparison queries (e.g., treatment-only or single-disease symptom queries).\n\n"

        "MISSING FACTS RULE:\n"
        "- If missing_facts are listed below: do not state or speculate about these missing facts; explicitly say what could not be found if relevant.\n\n"

        "VACCINE TIMINGS RULE:\n"
        "- For newborn vaccines in India, focus ONLY on the birth doses explicitly supported by official guidelines (BCG, OPV 0, Hepatitis B birth dose).\n"
        "- Do NOT list or state unverified subsequent vaccine schedules (such as rotavirus, IPV, DPT, PCV) unless those exact timings are explicitly present in the provided reference sources.\n\n"

        "SAFETY CLOSING:\n"
        "- End naturally: 'Of course, I'm not a substitute for your doctor — if things feel worse or you're unsure, please get checked out.'"
    )

    src_texts = []
    for idx, s in enumerate(sources_collected, 1):
        stype = s.get("source_type", "source")
        name = s.get("doc_name") or s.get("title") or "Guideline"
        status = s.get("http_status")
        snip = (s.get("snippet") or s.get("content_snippet") or "")[:800]
        if status and status >= 400:
            src_texts.append(f"Source [{idx}] ({stype}, HTTP {status} - snippet only): {name}\nSnippet: {snip}\nNOTE: Status {status}. Do NOT attribute beyond this snippet.")
        else:
            src_texts.append(f"Source [{idx}] ({stype}): {name}\nSnippet: {snip}")

    context_str = "\n\n".join(src_texts)
    missing_short = missing_facts[:3] if missing_facts else []
    missing_str = f"\nMissing Facts (do not state/speculate on these): {missing_short}" if missing_short else ""

    user_msg_content = f"User Query: {query}\n\nRetrieved Reference Sources:\n{context_str}{missing_str}"

    generator_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg_content}
    ]

    t_gen_start = time.time()
    gen_response = safe_chat_completion(
        client,
        active_model,
        messages=generator_messages,
        temperature=0.2,
        max_tokens=1000
    )
    dt_gen = round((time.time() - t_gen_start) * 1000, 2)
    timing_breakdown.append({"step": "Groq Call (Generation)", "duration_ms": dt_gen})

    generator_served_model = getattr(gen_response, "model", active_model)
    draft_answer = (gen_response.choices[0].message.content or "").strip()
    if not draft_answer:
        print("  [GENERATION RETRY] First generation attempt returned empty text. Retrying with fallback model...")
        gen_response = safe_chat_completion(
            client,
            "openai/gpt-oss-120b",
            messages=generator_messages,
            temperature=0.2,
            max_tokens=1000
        )
        generator_served_model = getattr(gen_response, "model", "openai/gpt-oss-120b")
        draft_answer = (gen_response.choices[0].message.content or "Information unavailable.").strip()

    print(f"  [GENERATOR MODEL] Served by: '{generator_served_model}'")

    # 4. AUDIT DRAFT
    t_audit_draft_start = time.time()
    draft_audit = verify_grounding(client, active_model, draft_answer, sources_collected,
                                   generator_served_model=generator_served_model)
    dt_audit_draft = round((time.time() - t_audit_draft_start) * 1000, 2)
    timing_breakdown.append({"step": "Groq Call (Draft Audit)", "duration_ms": dt_audit_draft})

    draft_unsupported = draft_audit.get("unsupported_sentences", [])
    draft_ungrounded = draft_audit.get("is_fully_grounded") is False

    # 5. CONSTRAINED REGENERATION (if draft failed audit)
    final_answer = draft_answer
    final_audit = draft_audit
    regen_fired = False

    if draft_ungrounded and draft_unsupported:
        print(f"  [REGEN TRIGGER] Draft audit found {len(draft_unsupported)} unsupported sentence(s). Regenerating...")
        t_regen_start = time.time()
        final_answer = _regenerate_constrained(
            client, active_model, query, draft_answer, sources_collected, draft_unsupported
        )
        dt_regen = round((time.time() - t_regen_start) * 1000, 2)
        timing_breakdown.append({"step": "Groq Call (Constrained Regeneration)", "duration_ms": dt_regen})
        regen_fired = True

        # 6. RE-AUDIT FINAL
        t_audit_final_start = time.time()
        final_audit = verify_grounding(client, active_model, final_answer, sources_collected,
                                       generator_served_model=generator_served_model)
        dt_audit_final = round((time.time() - t_audit_final_start) * 1000, 2)
        timing_breakdown.append({"step": "Groq Call (Final Audit)", "duration_ms": dt_audit_final})

    # 7. APPEND DOCTOR NOTE AFTER AUDIT (treatment queries from live sources)
    # This note is appended in code — not filterable by regen — for any query where
    # live sources discuss treatment/antibiotics/medication.
    is_treatment_query = any(k in query.lower() for k in ["treatment", "treat", "antibiotic", "medicine", "medication", "drug"])
    if is_treatment_query and fallback_fired:
        DOCTOR_NOTE = "A doctor chooses the treatment, especially in pregnancy or for young children."
        if DOCTOR_NOTE.lower() not in final_answer.lower():
            final_answer = final_answer.rstrip()
            # Insert before the final closing disclaimer if present
            closing = "Of course, I'm not a substitute for your doctor"
            if closing.lower() in final_answer.lower():
                idx_closing = final_answer.lower().rfind(closing.lower())
                final_answer = (
                    final_answer[:idx_closing].rstrip() + "\n\n"
                    + DOCTOR_NOTE + "\n\n"
                    + final_answer[idx_closing:]
                )
            else:
                final_answer += f"\n\n{DOCTOR_NOTE}"
            print(f"  [DOCTOR NOTE] Appended treatment authority note for treatment query.")

    # 8. OPTIONAL SYMPTOM LOGGING
    personal_keywords = ["i have", "my baby", "my child", "i am", "my fever", "suffering from", "pain in my"]
    if any(k in query.lower() for k in personal_keywords):
        bg_executor.submit(log_symptom, user_id, query, "general", triage_tag)

    total_duration = round((time.time() - overall_start) * 1000, 2)
    is_grounded_val = final_audit.get("is_fully_grounded")
    draft_ungrounded_rate = 1 if draft_ungrounded else 0

    trace = {
        "query": query,
        "retrieval_query": query,
        "is_comparison_query": is_comparison,
        "faiss_chunks": [
            {
                "chunk_id": s.get("chunk_id"),
                "doc_name": s.get("doc_name"),
                "section": s.get("section"),
                "score": float(s.get("score", 0.0)),
                "snippet_80": (s.get("chunk_text") or s.get("snippet", ""))[:80]
            }
            for s in (gate_result.get("valid_chunks", []) if gate_result else [])
        ],
        "max_ce_score": gate_result.get("max_ce_score") if gate_result else None,
        "chunk_ce_scores": gate_result.get("chunk_ce_scores", []) if gate_result else [],
        "gate_decision": {
            "passed": gate_result.get("passed") if gate_result else False,
            "skip_kb": gate_result.get("skip_kb", False),
            "reason": gate_result.get("reason") if gate_result else "No gate result"
        },
        "sufficiency_check": suff_result,
        "web_fetch_logs": [
            {
                "title": s.get("title"),
                "source_url": s.get("source_url"),
                "http_status": s.get("http_status"),
                "content_type": s.get("content_type"),
                "bytes_len": s.get("bytes_len"),
                "content_source": s.get("content_source"),
                "page_date": s.get("page_date"),
                "snippet": s.get("snippet")
            }
            for s in sources_collected if s.get("source_type") == "live_search"
        ],
        "draft_answer": draft_answer,
        "draft_audit": {
            "is_fully_grounded": draft_audit.get("is_fully_grounded"),
            "unsupported_sentences": draft_audit.get("unsupported_sentences", []),
            "audit_summary": draft_audit.get("audit_summary"),
            "auditor_model": draft_audit.get("auditor_model")
        },
        "regen_fired": regen_fired,
        "final_answer": final_answer,
        "final_audit": {
            "is_fully_grounded": final_audit.get("is_fully_grounded"),
            "unsupported_sentences": final_audit.get("unsupported_sentences", []),
            "audit_summary": final_audit.get("audit_summary"),
            "auditor_model": final_audit.get("auditor_model")
        },
        "draft_ungrounded_rate": draft_ungrounded_rate,
        "generator_messages": generator_messages,
        "audit_messages": final_audit.get("audit_messages", []),
        "timings_ms": {t["step"]: t["duration_ms"] for t in timing_breakdown}
    }

    print(f"[ORCHESTRATOR COMPLETE] Total Latency: {total_duration} ms | Draft Grounded: {draft_audit.get('is_fully_grounded')} | Final Grounded: {is_grounded_val} | Regen: {regen_fired}\n")
    return {
        "model_used": active_model,
        "answer": final_answer,
        "fallback_fired": fallback_fired,
        "is_grounded": is_grounded_val,
        "tools_used": ["search_advisories"] + (["web_search_health_authority"] if fallback_fired else []),
        "tool_call_trace": [],
        "triage_tag": triage_tag,
        "sources": sources_collected,
        "grounding_audit": final_audit,
        "gate_result": gate_result,
        "trace": trace,
        "timing_breakdown": timing_breakdown,
        "response_time_ms": total_duration
    }
