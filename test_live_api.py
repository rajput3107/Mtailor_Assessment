import requests
import base64
import json
from PIL import Image
import io

# Your Cerebrium endpoint and API key
API_ENDPOINT = "https://api.cerebrium.ai/v4/p-714e82e6/image-classifier/run"
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWM2Y2VmY2YwIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0MjcxMzc5fQ.5olvNRjH2e1092pyWCtIgbn0IiKX65ZubnlHptharnoexlGKancS7G_ITRZJcnVaoww61ji-tXvoj8fgdN1wGBMcqkG_TyB8caoPAXd6JenLxxEaXO_BdHCHgUp5oZMdRNBSZKTmq9VNC-o1de3NGXUIH-x9T7W45t7x8CpZzsOuW6nlIcDOAj3vLym7U238zNtPh5e7vfDWldz5Ogo9dZ9Mq3u-KJxexgh_pw79x2zyKRWquXRrfH3NrvWyAP5cIkjg4TfVbX4K2YBWaXc3N4VnoEgUehcO99mXHKYCsrorEA5J4YsLCi5nqPnkzQako-xBrlmXBPIYuyAutdGjAQ"

def _encode_image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image object to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG") # Use JPEG for smaller size and common compatibility
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_api_availability():
    """
    Test if the API is available and responds to a minimal valid request.
    Since the endpoint expects an 'image' attribute, this acts as a basic
    "ping" or availability check by sending a small dummy image.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("Testing API availability (sending minimal image request)...")
    
    # Create a tiny 1x1 black image as a dummy
    dummy_img = Image.new('RGB', (1, 1), color='black')
    dummy_image_base64 = _encode_image_to_base64(dummy_img)

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json={"image": dummy_image_base64}, timeout=10) # Added timeout
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        # Check if the response contains 'predictions' to confirm a successful model run
        if 'predictions' in response.json():
            print("✅ API is available and responded with predictions!")
            # Optionally print a snippet of the response for confirmation
            print(f"Response snippet: {json.dumps(response.json()['predictions'][:1], indent=2)}...")
            return True
        else:
            print(f"❌ API responded with status {response.status_code} but no 'predictions' in payload.")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return False

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred: {http_err} - Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    except requests.exceptions.ConnectionError as conn_err:
        print(f"❌ Connection error occurred: {conn_err}")
        return False
    except requests.exceptions.Timeout as timeout_err:
        print(f"❌ Request timed out: {timeout_err}")
        return False
    except requests.exceptions.RequestException as req_err:
        print(f"❌ An unexpected request error occurred: {req_err}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Failed to decode JSON from response. Response text: {response.text}")
        return False

def test_image_classification():
    """Test image classification with a slightly larger, colored image."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a test image (e.g., a blue square)
    print("\nCreating test image (224x224 blue square)...")
    test_img = Image.new('RGB', (224, 224), color='blue')
    image_base64 = _encode_image_to_base64(test_img)
    
    # Make request
    print("Sending classification request...")
    try:
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json={"image": image_base64},
            timeout=30 # Increased timeout for actual inference
        )
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        if 'predictions' in result:
            print("✅ Classification successful!")
            print("\nTop 5 Predictions:")
            print("-" * 50)
            # Ensure we only print up to 5 predictions if fewer are returned
            for i, pred in enumerate(result['predictions'][:5], 1): 
                print(f"{i}. {pred['class']}: {pred['confidence']:.2%}")
            return True
        else:
            print(f"❌ Classification failed: 'predictions' key not found in response.")
            print(f"Response: {json.dumps(result, indent=2)}")
            return False

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error occurred during classification: {http_err} - Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    except requests.exceptions.ConnectionError as conn_err:
        print(f"❌ Connection error occurred during classification: {conn_err}")
        return False
    except requests.exceptions.Timeout as timeout_err:
        print(f"❌ Classification request timed out: {timeout_err}")
        return False
    except requests.exceptions.RequestException as req_err:
        print(f"❌ An unexpected request error occurred during classification: {req_err}")
        return False
    except json.JSONDecodeError:
        print(f"❌ Failed to decode JSON from classification response. Response text: {response.text}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Testing Live Cerebrium Image Classifier API")
    print("=" * 60)
    print(f"Endpoint: {API_ENDPOINT}")
    print("=" * 60)
    
    # Run tests
    if test_api_availability(): # Renamed for clarity
        print("\n" + "=" * 60)
        print("Proceeding to full image classification test...")
        print("=" * 60)
        test_image_classification()
    else:
        print("\n" + "=" * 60)
        print("API availability check failed. Skipping full classification test.")
        print("=" * 60)