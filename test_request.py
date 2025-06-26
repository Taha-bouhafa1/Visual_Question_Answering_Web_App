import requests

url = 'http://127.0.0.1:5000/predict'

image_path = r'C:\Users\Mac Planet\Documents\PythonProject\vqa_web_app\backend\data\6alkr00kmx_Asian_Elephants_WW252891.jpg'
question = "What is in the image?"

with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    data = {'question': question}
    response = requests.post(url, files=files, data=data)

if response.ok:
    print("Response from server:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
