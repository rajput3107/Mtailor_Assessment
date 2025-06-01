import requests
import base64
import json
from PIL import Image
import io

# REPLACE THESE VALUES:
API_ENDPOINT = "https://api.cortex.cerebrium.ai/v4/p-c6cefc0/image-classifier/predict"  # Using your project ID
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWM2Y2VmY2YwIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0MjcxMzc5fQ.5olvNRjH2e1092pyWCtIgbn0IiKX65ZubnlHptharnoexlGKancS7G_ITRZJcnVaoww61ji-tXvoj8fgdN1wGBMcqkG_TyB8caoPAXd6JenLxxEaXO_BdHCHgUp5oZMdRNBSZKTmq9VNC-o1de3NGXUIH-x9T7W45t7x8CpXzsOuW6nlIcDOAj3vLym7U238zNtPh5e7vfDWldn5Ogo9dZ9Mq3u-KJxexgh_pw79x2zyKRWquXRrfH3NrvWyAP5cIkjg4TfVbX4K2YBWaXc3N4VnoEgUehcO99mXHKYCsrorEA5J4YsLCi5nqPnkzQako-xBrlmXBPIYuyAutdGjAQ"  # Your inference token

def classify_image_via_api(image_path):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    with open(image_path, "rb") as image_file:
        encoded_image_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Match the payload format expected by your main.py
    payload = {"image": encoded_image_string}

    print(f"Sending request to {API_ENDPOINT}...")
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        print("--- API Response ---")
        print(json.dumps(result, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if 'response' in locals():
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")

if __name__ == "__main__":
    # Create a dummy image file for testing
    dummy_img = Image.new('RGB', (224, 224), color='red')
    dummy_img.save("dummy_test_image.jpg")

    classify_image_via_api("dummy_test_image.jpg")