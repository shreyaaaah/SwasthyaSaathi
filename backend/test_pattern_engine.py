import sys
import os
from datetime import datetime, timedelta

# Ensure stdout and stderr handle UTF-8 characters on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.db import SessionLocal, init_db
from app.models import SymptomLog
from app.agent.pattern_engine import detect_symptom_patterns

def test_pattern_detection():
    init_db()
    db = SessionLocal()
    test_user = "pattern_test_user_99"

    print("=" * 70)
    print(f"TESTING LONGITUDINAL PATTERN ENGINE FOR USER: {test_user}")
    print("=" * 70)

    # Clean up existing test entries
    db.query(SymptomLog).filter(SymptomLog.user_id == test_user).delete()
    db.commit()

    # Case 1: Initial query (no pattern yet)
    print("\n--- TEST CASE 1: Single Log (No Pattern Expected) ---")
    log1 = SymptomLog(
        user_id=test_user,
        query_text="I have a mild dry cough",
        topic="tuberculosis",
        triage_tag="GENERAL_INFO",
        created_at=datetime.utcnow() - timedelta(days=6)
    )
    db.add(log1)
    db.commit()

    res1 = detect_symptom_patterns(test_user, days_window=14)
    print("Result 1:", res1)
    assert not res1["pattern_detected"], "Expected pattern_detected = False for 1 entry"

    # Case 2: Second cough query (Recurring Pattern Expected)
    print("\n--- TEST CASE 2: Second Cough Log 3 Days Later (Recurring Pattern) ---")
    log2 = SymptomLog(
        user_id=test_user,
        query_text="My cough is still there after 3 days",
        topic="tuberculosis",
        triage_tag="CONSULT_SOON",
        created_at=datetime.utcnow() - timedelta(days=3)
    )
    db.add(log2)
    db.commit()

    res2 = detect_symptom_patterns(test_user, days_window=14)
    print("Result 2:", res2)
    assert res2["pattern_detected"], "Expected pattern_detected = True for 2 cough entries"

    # Case 3: Third cough query with escalating severity (Escalating Pattern Expected)
    print("\n--- TEST CASE 3: Third Cough Log with Severe Symptoms (Escalating Pattern) ---")
    log3 = SymptomLog(
        user_id=test_user,
        query_text="Persistent cough for 2 weeks with blood in sputum",
        topic="tuberculosis",
        triage_tag="EMERGENCY",
        created_at=datetime.utcnow()
    )
    db.add(log3)
    db.commit()

    res3 = detect_symptom_patterns(test_user, days_window=14)
    print("Result 3:", res3)
    assert res3["pattern_detected"], "Expected pattern_detected = True"
    assert res3["pattern_type"] == "escalating", "Expected pattern_type = 'escalating'"

    print("\n" + "=" * 70)
    print("SUCCESS: Longitudinal Pattern Engine correctly flagged recurring & escalating symptom trends!")
    print("=" * 70)

if __name__ == "__main__":
    test_pattern_detection()
