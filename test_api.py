"""Direct API test — bypasses Swagger UI entirely."""

import httpx

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = "http" + "://" + HOST + ":" + str(PORT)

# Health
print("Health:", httpx.get(f"{BASE_URL}/health").json())

# Single-file endpoint
with open("docs/long_article.txt", "rb") as f:
    files = {"file": ("long_article.txt", f, "text/plain")}
    r = httpx.post(f"{BASE_URL}/summarize-one", files=files, timeout=180)
print("summarize-one status:", r.status_code)
print("Result:", r.json())

# Multi-file endpoint
with open("docs/doc1.txt", "rb") as f1, open("docs/doc2.txt", "rb") as f2:
    files = [
        ("files", ("doc1.txt", f1, "text/plain")),
        ("files", ("doc2.txt", f2, "text/plain")),
    ]
    r = httpx.post(f"{BASE_URL}/summarize", files=files, timeout=180)
print("summarize status:", r.status_code)
for item in r.json():
    print(item["source"], ":", item["summary"][:80])