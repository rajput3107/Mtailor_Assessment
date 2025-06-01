"""
Test script for deployed Cerebrium model
"""
import requests
import base64
import json
import argparse
import os
import time
from typing import Dict, List
import concurrent.futures
from PIL import Image
import io

class CerebriumTester:
    """Test suite for Cerebrium deployment"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def test_single_prediction(self, image_path: str) -> Dict:
        """Test single image prediction"""
        print(f"\nTesting prediction for: {image_path}")
        
        # Prepare request
        image_base64 = self.encode_image(image_path)
        payload = {
            "image": image_base64
        }
        
        # Make request
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/predict",
            headers=self.headers,
            json=payload
        )
        end_time = time.time()
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print(f"✓ Prediction successful!")
                print(f"  - Class ID: {result['predicted_class_id']}")
                print(f"  - Confidence: {result['confidence']:.4f}")
                print(f"  - Response time: {end_time - start_time:.2f}s")
                print(f"  - Inference time: {result['timing']['inference_time']:.3f}s")
                return result
            else:
                print(f"✗ Prediction failed: {result.get('error', 'Unknown error')}")
                return None
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print("\nTesting health check...")
        
        response = requests.post(
            f"{self.api_url}/health_check",
            headers=self.headers,
            json={}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "healthy":
                print("✓ Health check passed")
                return True
        
        print("✗ Health check failed")
        return False
    
    def test_concurrent_requests(self, image_paths: List[str], num_concurrent: int = 5):
        """Test concurrent requests to check scalability"""
        print(f"\nTesting {num_concurrent} concurrent requests...")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(self.test_single_prediction, path)
                for path in image_paths[:num_concurrent]
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        end_time = time.time()
        
        print(f"\nConcurrent test summary:")
        print(f"  - Total time: {end_time - start_time:.2f}s")
        print(f"  - Successful requests: {len(results)}/{num_concurrent}")
        
        if results:
            avg_inference = sum(r['timing']['inference_time'] for r in results) / len(results)
            print(f"  - Average inference time: {avg_inference:.3f}s")
    
    def test_error_handling(self):
        """Test error handling"""
        print("\nTesting error handling...")
        
        # Test with invalid image
        print("Testing with invalid base64...")
        response = requests.post(
            f"{self.api_url}/predict",
            headers=self.headers,
            json={"image": "invalid_base64"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "error":
                print("✓ Invalid base64 handled correctly")
            else:
                print("✗ Invalid base64 not handled properly")
        
        # Test with missing image
        print("Testing with missing image...")
        response = requests.post(
            f"{self.api_url}/predict",
            headers=self.headers,
            json={}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "error":
                print("✓ Missing image handled correctly")
            else:
                print("✗ Missing image not handled properly")
    
    def test_performance_benchmark(self, image_path: str, num_requests: int = 10):
        """Benchmark model performance"""
        print(f"\nRunning performance benchmark ({num_requests} requests)...")
        
        response_times = []
        inference_times = []
        
        for i in range(num_requests):
            start_time = time.time()
            result = self.test_single_prediction(image_path)
            end_time = time.time()
            
            if result:
                response_times.append(end_time - start_time)
                inference_times.append(result['timing']['inference_time'])
            
            # Small delay between requests
            time.sleep(0.1)
        
        if response_times:
            print(f"\nPerformance Summary:")
            print(f"  - Average response time: {np.mean(response_times):.3f}s")
            print(f"  - Min response time: {np.min(response_times):.3f}s")
            print(f"  - Max response time: {np.max(response_times):.3f}s")
            print(f"  - Average inference time: {np.mean(inference_times):.3f}s")
            print(f"  - 95th percentile response: {np.percentile(response_times, 95):.3f}s")
    
    def run_preset_tests(self):
        """Run preset custom tests"""
        print("\n" + "="*50)
        print("Running Preset Test Suite")
        print("="*50)
        
        # Expected test results
        test_images = {
            "test_images/n01440764_tench.JPEG": 0,  # Expected class 0
            "test_images/n01667114_mud_turtle.JPEG": 35  # Expected class 35
        }
        
        # 1. Health check
        if not self.test_health_check():
            print("Health check failed, aborting tests")
            return
        
        # 2. Test known images
        print("\nTesting known images...")
        for image_path, expected_class in test_images.items():
            if os.path.exists(image_path):
                result = self.test_single_prediction(image_path)
                if result:
                    predicted_class = result['predicted_class_id']
                    if predicted_class == expected_class:
                        print(f"✓ Correct prediction for {os.path.basename(image_path)}")
                    else:
                        print(f"✗ Incorrect prediction: expected {expected_class}, got {predicted_class}")
            else:
                print(f"✗ Test image not found: {image_path}")
        
        # 3. Error handling
        self.test_error_handling()
        
        # 4. Concurrent requests
        existing_images = [p for p in test_images.keys() if os.path.exists(p)]
        if existing_images:
            self.test_concurrent_requests(existing_images * 3, num_concurrent=5)
        
        # 5. Performance benchmark
        if existing_images:
            self.test_performance_benchmark(existing_images[0], num_requests=10)
        
        print("\n" + "="*50)
        print("Test Suite Completed")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Test Cerebrium deployment')
    parser.add_argument(
        '--api-url',
        type=str,
        required=True,
        help='Cerebrium API URL (e.g., https://api.cerebrium.ai/v4/p-xxxxx/model-name)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='Cerebrium API key'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image for single prediction test'
    )
    parser.add_argument(
        '--run-tests',
        action='store_true',
        help='Run preset custom tests'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = CerebriumTester(args.api_url, args.api_key)
    
    if args.run_tests:
        # Run full test suite
        tester.run_preset_tests()
    elif args.image:
        # Test single image
        tester.test_single_prediction(args.image)
    else:
        # Default: run basic tests
        tester.test_health_check()
        print("\nUse --image <path> for single prediction or --run-tests for full test suite")


if __name__ == "__main__":
    import numpy as np  # Import here for stats
    main()
                