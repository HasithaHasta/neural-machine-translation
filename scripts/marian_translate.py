import json
import requests

API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-hi"
headers = {"Authorization": "Bearer api_wCIMGGqoNgSsaTxlWxzCUJqRawaXygigtR"}

def query(payload):
	data = json.dumps(payload)
	response = requests.request("POST", API_URL, headers=headers, data=data)
	return json.loads(response.content.decode("utf-8"))

# inp = str(input())


# data = query(
#     {
#         "inputs": inp,
#     }
# )
# print(data[0]['translation_text'])

def gen_translation(inp):
    data = query({"inputs": inp})
    return data[0]['translation_text']

# inp = 'This is a sample sentence for testing.'
# print(gen_translation(inp))