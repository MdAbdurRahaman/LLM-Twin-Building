import requests
import json

print("--- Testing Live Gradio Web Server HTTP Endpoint ---")

BASE_URL = "http://localhost:7860"

try:
    # 1. Test Web Server Root Response
    r = requests.get(BASE_URL, timeout=5)
    print(f"[HTTP TEST] Root status code: {r.status_code}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    
    # 2. Test Gradio Info Config Endpoint
    config_url = f"{BASE_URL}/config"
    rc = requests.get(config_url, timeout=5)
    if rc.status_code == 200:
        print("[HTTP TEST] Gradio Config Endpoint responding cleanly!")
        
    print("[SUCCESS] Live Web UI server is UP and serving cleanly on http://localhost:7860!")

except Exception as e:
    print(f"[FAILED] HTTP Test Failed: {e}")
