import requests
import time
import random

URL = "http://localhost:5000/invocations"

while True:
    data = {
        "dataframe_records": [
            {"feature1": random.random(), "feature2": random.random()}
        ]
    }
    start = time.time()
    try:
        r = requests.post(URL, json=data)
        latency = time.time() - start
        print(f"Status: {r.status_code}, Latency: {latency:.2f}s")
    except:
        print("Request failed")
    time.sleep(1)
