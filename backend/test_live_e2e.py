import requests
import json
import time

BASE_URL = "http://localhost:8000"
USER_ID = "live_test_user_77"

def test_live_e2e():
    print("=" * 80)
    print("RUNNING LIVE END-TO-END VERIFICATION AGAINST http://localhost:8000")
    print("=" * 80)

    # 1. Health check
    h_res = requests.get(f"{BASE_URL}/health")
    print("\n1. Health Check Response:", h_res.json())

    # 2. Query 1: SELF_CARE / GENERAL_INFO
    print("\n" + "-" * 80)
    print("2. SENDING QUERY 1: 'does turmeric cure diabetes'")
    q1_start = time.time()
    res1 = requests.post(f"{BASE_URL}/chat", json={"query": "does turmeric cure diabetes", "user_id": USER_ID})
    q1_dt = round((time.time() - q1_start) * 1000, 2)
    data1 = res1.json()
    print(f"Status Code: {res1.status_code} | Time: {q1_dt} ms")
    print("JSON Response 1:")
    print(json.dumps(data1, indent=2))

    # 3. Query 2: EMERGENCY
    print("\n" + "-" * 80)
    print("3. SENDING QUERY 2: 'I'm having severe chest pain and can't breathe'")
    q2_start = time.time()
    res2 = requests.post(f"{BASE_URL}/chat", json={"query": "I'm having severe chest pain and can't breathe", "user_id": USER_ID})
    q2_dt = round((time.time() - q2_start) * 1000, 2)
    data2 = res2.json()
    print(f"Status Code: {res2.status_code} | Time: {q2_dt} ms")
    print("JSON Response 2:")
    print(json.dumps(data2, indent=2))

    # 4. GET /api/history/{user_id}
    print("\n" + "-" * 80)
    print(f"4. FETCHING GET /api/history/{USER_ID}")
    hist_res = requests.get(f"{BASE_URL}/api/history/{USER_ID}")
    hist_data = hist_res.json()
    print(f"Status Code: {hist_res.status_code}")
    print("Logged History Items Count:", len(hist_data))
    print(json.dumps(hist_data, indent=2))

    # 5. GET /api/patterns/{user_id}
    print("\n" + "-" * 80)
    print(f"5. FETCHING GET /api/patterns/{USER_ID}")
    pat_res = requests.get(f"{BASE_URL}/api/patterns/{USER_ID}")
    pat_data = pat_res.json()
    print(f"Status Code: {pat_res.status_code}")
    print("Pattern Detection Result:")
    print(json.dumps(pat_data, indent=2))

if __name__ == "__main__":
    test_live_e2e()
