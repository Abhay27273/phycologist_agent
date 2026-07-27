import requests

# USE 127.0.0.1 instead of localhost
url = "http://127.0.0.1:8000/api/v1/chat"

data = {
    "user_id": "test_user_01",
    "session_id": "session_A",
    "message": "I feel really overwhelmed with work lately. I can't sleep."
}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print(f"Connection Failed: {e}")