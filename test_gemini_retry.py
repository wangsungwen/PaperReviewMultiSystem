
import json
import time
import random

# Mocking the dictionary structure of Gemini's 429 response
mock_429_response = {
    "error": {
        "code": 429,
        "message": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash-lite. Please retry in 29.723877525s.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "29.723877525s"
            }
        ]
    }
}

def simulate_retry_logic(response_json):
    attempt = 0
    max_retries = 5
    
    # Logic extracted from interface.py
    wait_time = (2 ** attempt) * 5 + random.uniform(0, 1)  
    
    try:
        response_data = response_json
        error_data = response_data.get("error", {})
        retry_info = error_data.get("details", [])
        
        found_delay = False
        for detail in retry_info:
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                retry_delay_str = detail.get("retryDelay", "5s")
                wait_time = float(retry_delay_str.rstrip('s'))
                found_delay = True
                print(f"Found explicit retry delay: {wait_time}s")
                break
        
        if not found_delay:
            print(f"No explicit delay found, using default/exponential: {wait_time}s")

        if error_data.get("status") == "RESOURCE_EXHAUSTED":
            error_msg = error_data.get("message", "")
            if "limit: 0" in error_msg:
                return f"【Gemini 額度耗盡】：您的每日 API 配額已用完。請更換 API Key 或明日再試。\n詳細訊息：{error_msg}"
    except Exception as e:
        print(f"Error during parsing: {e}")
        pass
    
    return f"Will retry in {wait_time:.2f}s"

print("--- Test 1: Full Quota Exhausted (limit: 0) ---")
result1 = simulate_retry_logic(mock_429_response)
print(result1)

print("\n--- Test 2: Transient Rate Limit (limit > 0) ---")
mock_transient_response = {
    "error": {
        "code": 429,
        "message": "Resource has been exhausted (e.g. check quota).",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "5.5s"
            }
        ]
    }
}
result2 = simulate_retry_logic(mock_transient_response)
print(result2)

print("\n--- Test 3: No Detail (Fallback to exponential) ---")
mock_no_detail_response = {
    "error": {
        "code": 429,
        "message": "Too Many Requests",
        "status": "UNAVAILABLE"
    }
}
result3 = simulate_retry_logic(mock_no_detail_response)
print(result3)
