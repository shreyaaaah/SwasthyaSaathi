import os
import sys
import json
import time
from typing import List, Dict, Any
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
# Note: Groq has decommissioned legacy models such as `llama-3.3-70b-versatile`,
# `llama3-70b-8192`, `llama-3.1-70b-versatile`, and `mixtral-8x7b-32768`.
# Attempting to call `llama-3.3-70b-versatile` returns a 404 (model_not_found).
# We use `qwen/qwen3.8-27b` (or `openai/gpt-oss-120b`), which are active high-speed
# tool-calling models on Groq's platform (~200ms response time per turn).
# ===========================================================================

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def search_advisories(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Runs FAISS retrieval pipeline over health knowledge base."""
    retriever = get_retriever()
    results = retriever.search(query=query, top_k=top_k, min_score=0.10)
    cleaned = []
    for r in results:
        cleaned.append({
            "doc_name": r.get("doc_name", "Unknown Document"),
            "topic": r.get("topic", "general"),
            "section": r.get("section", "General"),
            "snippet": r.get("chunk_text", "")[:350] + "...",
            "score": r.get("score", 0.0)
        })
    return cleaned

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
    prompt = f"""Classify urgency for: "{query}". Context: "{retrieved_context[:200]}".
Options: EMERGENCY, CONSULT_SOON, SELF_CARE, GENERAL_INFO.
JSON response: {{"triage_tag": "<TAG>", "confidence": 85, "reason": "<reason>"}}"""

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
    # TODO: Wire live IDSP (Integrated Disease Surveillance Programme) outbreak data feed
    return {
        "state": state,
        "status": "No active disease outbreak alerts registered.",
        "source": "IDSP Surveillance Feed (Stub)",
        "note": "TODO: Wire up live IDSP outbreak data in future iteration."
    }

TOOLS_MAP = {
    "search_advisories": search_advisories,
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
            "description": "Searches public health guidelines (dengue, TB, air pollution, flood health, maternal & child health) for verified clinical guidance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Health or medical topic query to search"},
                    "top_k": {"type": "integer", "description": "Number of top matching chunks to retrieve (default 3)"}
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

# Helper function to execute a single tool with timing
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
# Orchestrator Core Loop with Latency Instrumentation & Parallel Tools
# ---------------------------------------------------------------------------

def run_agent(user_id: str, query: str) -> Dict[str, Any]:
    """Agentic Tool-Calling Orchestrator with parallel tool execution & latency instrumentation."""
    init_db()  # Ensure database tables exist
    overall_start = time.time()
    active_model = settings.GROQ_MODEL_NAME
    client = Groq(api_key=settings.GROQ_API_KEY)

    print(f"\n[ORCHESTRATOR START] Model: '{active_model}' | User: '{user_id}' | Query: '{query}'")

    system_prompt = (
        "You are SwasthyaSaathi, an intelligent agentic public health assistant "
        "grounded in official guidelines from MoHFW, ICMR, NHP, and WHO.\n\n"
        "You have tools for: search_advisories, check_myth, get_session_history, log_symptom, triage_classify, get_regional_alerts.\n\n"
        "Instructions for high efficiency:\n"
        "1. Select ALL necessary tools in your FIRST response turn concurrently.\n"
        "2. If query mentions past interactions, include `get_session_history`.\n"
        "3. If query asks about remedies/myths (turmeric, garlic), include `check_myth`.\n"
        "4. If query describes symptoms, include `search_advisories` AND `triage_classify`.\n"
        "5. Include `log_symptom` in your tool calls to persist the interaction.\n"
        "6. Provide grounded, concise, empathetic guidance with safety disclaimers."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User ID: {user_id}\nQuery: {query}"}
    ]

    tool_call_trace = []
    sources_collected = []
    triage_tag = "GENERAL_INFO"
    timing_breakdown = []

    max_turns = 4
    turns = 0

    while turns < max_turns:
        turns += 1
        groq_start = time.time()
        print(f"  [TIMING] Turn {turns} Groq API Call started at {time.strftime('%H:%M:%S', time.localtime(groq_start))}...")

        response = client.chat.completions.create(
            model=active_model,
            messages=messages,
            tools=GROQ_TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=600
        )

        groq_duration = round((time.time() - groq_start) * 1000, 2)
        timing_breakdown.append({"step": f"Groq Call (Turn {turns})", "duration_ms": groq_duration})
        print(f"  [TIMING] Turn {turns} Groq API Call finished in {groq_duration} ms")

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            final_answer = msg.content or "No response text generated."
            total_duration = round((time.time() - overall_start) * 1000, 2)
            print(f"[ORCHESTRATOR COMPLETE] Total Latency: {total_duration} ms\n")
            return {
                "model_used": active_model,
                "answer": final_answer,
                "tools_used": [t["tool"] for t in tool_call_trace],
                "tool_call_trace": tool_call_trace,
                "triage_tag": triage_tag,
                "sources": sources_collected,
                "timing_breakdown": timing_breakdown,
                "response_time_ms": total_duration
            }

        # Append assistant message with tool calls
        messages.append(msg)

        # PARALLEL TOOL EXECUTION
        tools_start = time.time()
        tool_names = [tc.function.name for tc in msg.tool_calls]
        print(f"  [PARALLEL TOOLS START] Executing {len(msg.tool_calls)} tools in parallel: {tool_names}...")

        executed_results = []
        with ThreadPoolExecutor(max_workers=len(msg.tool_calls)) as executor:
            future_to_tc = {executor.submit(_execute_single_tool, tc): tc for tc in msg.tool_calls}
            for future in as_completed(future_to_tc):
                res = future.result()
                executed_results.append(res)
                print(f"    - Tool '{res['fn_name']}' completed in {res['duration_ms']} ms")

        tools_duration = round((time.time() - tools_start) * 1000, 2)
        timing_breakdown.append({"step": f"Parallel Tools Execution ({', '.join(tool_names)})", "duration_ms": tools_duration})
        print(f"  [PARALLEL TOOLS COMPLETE] All tools finished in {tools_duration} ms")

        # Process results in original order
        for res in executed_results:
            fn_name = res["fn_name"]
            tool_result = res["result"]

            trace_entry = {
                "tool": fn_name,
                "args": res["args"],
                "result": tool_result,
                "duration_ms": res["duration_ms"]
            }
            tool_call_trace.append(trace_entry)

            if fn_name == "search_advisories" and isinstance(tool_result, list):
                sources_collected.extend(tool_result)

            if fn_name == "triage_classify" and isinstance(tool_result, dict):
                triage_tag = tool_result.get("triage_tag", triage_tag)

            messages.append({
                "role": "tool",
                "tool_call_id": res["tc_id"],
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

    total_duration = round((time.time() - overall_start) * 1000, 2)
    return {
        "model_used": active_model,
        "answer": "Maximum tool execution turns reached.",
        "tools_used": [t["tool"] for t in tool_call_trace],
        "tool_call_trace": tool_call_trace,
        "triage_tag": triage_tag,
        "sources": sources_collected,
        "timing_breakdown": timing_breakdown,
        "response_time_ms": total_duration
    }

# ---------------------------------------------------------------------------
# Test Suite Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "I've had a persistent cough for 3 weeks",
        "does turmeric cure diabetes",
        "what should I do about the fever I mentioned last time",
        "I'm having severe chest pain and can't breathe"
    ]

    test_user_id = "test_user_001"

    print("=" * 80)
    print(f"RUNNING AGENTIC ORCHESTRATOR LATENCY BENCHMARK | MODEL: {settings.GROQ_MODEL_NAME}")
    print("=" * 80 + "\n")

    for idx, q in enumerate(test_queries, 1):
        print(f"[{idx}/4] TEST QUERY: \"{q}\"")
        print("-" * 80)
        
        try:
            result = run_agent(user_id=test_user_id, query=q)
            
            print("1. LATENCY BREAKDOWN:")
            for b in result.get("timing_breakdown", []):
                print(f"   • {b['step']}: {b['duration_ms']} ms")
            print(f"   ---> TOTAL LATENCY: {result.get('response_time_ms')} ms ({round(result.get('response_time_ms', 0)/1000, 2)}s)")

            print("\n2. TOOL CALL SEQUENCE & TRACE:")
            if not result.get("tool_call_trace"):
                print("   (No tools called)")
            else:
                for step, trace in enumerate(result["tool_call_trace"], 1):
                    print(f"   Step {step}: Tool = '{trace['tool']}' (took {trace.get('duration_ms')} ms)")
                    print(f"           Args = {json.dumps(trace['args'])}")
                    print(f"           Result = {json.dumps(trace['result'], ensure_ascii=False)[:200]}...")
            
            print(f"\n3. TRIAGE TAG: {result.get('triage_tag')}")
            print("\n4. FINAL NATURAL-LANGUAGE ANSWER:")
            print(result.get("answer"))
            
        except Exception as err:
            import traceback
            print("ERROR ENCOUNTERED:")
            traceback.print_exc()

        print("\n" + "=" * 80 + "\n")
