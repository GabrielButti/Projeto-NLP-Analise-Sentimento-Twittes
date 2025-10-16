import requests

API_URL = "http://localhost:8000"
text = "I love it!"

resp = requests.post(f"{API_URL}/analise/", json={"texto": text})
print("Status code:", resp.status_code)
print("Response JSON:", resp.json())
