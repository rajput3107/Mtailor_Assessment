# debug_health.py
import requests
import json

def test_health_check():
    """Test the health check endpoint with detailed debugging"""
    base_url = "http://localhost:5000"  # or your actual URL
    
    print("Testing health check endpoint...")
    
    try:
        # Try the health endpoint
        response = requests.get(f"{base_url}/health")
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"JSON response: {json.dumps(data, indent=2)}")
            except:
                print("Response is not JSON")
        
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to the server. Is it running?")
        return False
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    # First, make sure the server is running
    print("Make sure your server is running (python main.py or cerebrium deploy)")
    print("-" * 50)
    
    success = test_health_check()
    print("-" * 50)
    print(f"Health check {'PASSED' if success else 'FAILED'}")