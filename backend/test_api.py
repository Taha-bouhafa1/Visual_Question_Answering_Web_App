import requests

url = "http://127.0.0.1:5000/predict"
image_path = "data/6alkr00kmx_Asian_Elephants_WW252891.jpg"
question = "What is in the picture?"

with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    data = {"question": question}
    response = requests.post(url, files=files, data=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
