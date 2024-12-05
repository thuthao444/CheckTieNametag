import requests

# Test checktie API
tie_url = "http://127.0.0.1:8000/tie"
with open("./sample5.png", "rb") as f:
    tie_response = requests.post(tie_url, files={"file": f})
print("Checktie response:", tie_response.json())

# Test checknametag API
nametag_url = "http://127.0.0.1:8000/nametag"
with open("./sample5.png", "rb") as f:
    nametag_response = requests.post(nametag_url, files={"file": f})
print("Checknametag response:", nametag_response.json())
