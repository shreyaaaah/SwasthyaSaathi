import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.dirname(APP_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.db import SessionLocal
from app.models import SymptomLog

TRIAGE_SEVERITY = {
    "GENERAL_INFO": 1,
    "SELF_CARE": 2,
    "CONSULT_SOON": 3,
    "EMERGENCY": 4
}

def detect_symptom_patterns(user_id: str, days_window: int = 14) -> Dict[str, Any]:
    """
    Rule-based longitudinal pattern engine.
    Analyzes user's symptom_logs history over `days_window` (default 14 days)
    to detect recurring symptoms or escalating urgency trends.
    """
    db = SessionLocal()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_window)
        logs = (
            db.query(SymptomLog)
            .filter(SymptomLog.user_id == user_id, SymptomLog.created_at >= cutoff_date)
            .order_by(SymptomLog.created_at.asc())
            .all()
        )

        if len(logs) < 2:
            return {
                "pattern_detected": False,
                "pattern_type": "none",
                "description": "No recurring or escalating symptom patterns detected.",
                "recommended_action": "Continue routine health monitoring.",
                "occurrences_count": len(logs),
                "latest_triage": logs[-1].triage_tag if logs else None,
                "topic": logs[-1].topic if logs else None
            }

        # Group logs by topic / symptom keyword
        topic_groups: Dict[str, List[SymptomLog]] = {}
        for l in logs:
            topic_key = (l.topic or "general").lower()

            # Group related respiratory / fever / symptom terms
            q_lower = l.query_text.lower()
            if "cough" in q_lower or "tb" in q_lower or "tuberculosis" in q_lower:
                topic_key = "tuberculosis_cough"
            elif "fever" in q_lower or "dengue" in q_lower:
                topic_key = "fever_dengue"
            elif "chest pain" in q_lower or "breathe" in q_lower:
                topic_key = "cardio_respiratory"

            if topic_key not in topic_groups:
                topic_groups[topic_key] = []
            topic_groups[topic_key].append(l)

        # 1. Check for Escalating Urgency Patterns
        for topic_key, group in topic_groups.items():
            if len(group) >= 2:
                severities = [TRIAGE_SEVERITY.get(item.triage_tag, 1) for item in group]
                # Check if severity strictly increased or reached EMERGENCY
                if (severities[-1] > severities[0]) or ("EMERGENCY" in [item.triage_tag for item in group]):
                    latest_log = group[-1]
                    topic_display = topic_key.replace("_", " ").title()
                    return {
                        "pattern_detected": True,
                        "pattern_type": "escalating",
                        "description": f"⚠️ Escalating symptom urgency detected for '{topic_display}' ({group[0].triage_tag} ➔ {latest_log.triage_tag}) across {len(group)} interactions in past {days_window} days.",
                        "recommended_action": "Urgent clinical evaluation at a medical center is strongly advised due to escalating symptom severity.",
                        "occurrences_count": len(group),
                        "latest_triage": latest_log.triage_tag,
                        "topic": topic_key
                    }

        # 2. Check for Recurring Symptoms (2+ occurrences in window)
        for topic_key, group in topic_groups.items():
            if len(group) >= 2:
                latest_log = group[-1]
                topic_display = topic_key.replace("_", " ").title()
                return {
                    "pattern_detected": True,
                    "pattern_type": "recurring",
                    "description": f"🔁 Recurring symptom detected: '{topic_display}' logged {len(group)} times in the last {days_window} days.",
                    "recommended_action": f"Persistent or recurring {topic_display} symptoms warrant professional medical evaluation at your nearest health center.",
                    "occurrences_count": len(group),
                    "latest_triage": latest_log.triage_tag,
                    "topic": topic_key
                }

        return {
            "pattern_detected": False,
            "pattern_type": "none",
            "description": "No recurring or escalating symptom patterns detected.",
            "recommended_action": "Continue routine health monitoring.",
            "occurrences_count": len(logs),
            "latest_triage": logs[-1].triage_tag,
            "topic": logs[-1].topic
        }

    except Exception as e:
        return {
            "pattern_detected": False,
            "pattern_type": "error",
            "description": f"Error running pattern detection: {str(e)}",
            "recommended_action": "Check system logs.",
            "occurrences_count": 0,
            "latest_triage": None,
            "topic": None
        }
    finally:
        db.close()
