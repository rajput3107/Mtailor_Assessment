import base64
import json
from PIL import Image
import io
import sys
import os

# Force reload of main module to avoid caching issues
if 'main' in sys.modules:
    del sys.modules['main']

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_health_check():
    """Test the health check by calling run() with no parameters"""
    print("Testing health check...")
    
    try:
        # Import main module
        import main
        
        # Call run with no parameters
        result = main.run()
        
        print(f"Health check result: {json.dumps(result, indent=2)}")
        return result.get('status_code') == 200
    except Exception as e:
        print(f"Error during health check: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction(image_path):
    """Test prediction with an image"""
    print(f"\nTesting prediction with image: {image_path}")
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            # Create a test image
            print("Creating test image...")
            test_img = Image.new('RGB', (224, 224), color='red')
            test_img.save(image_path)
        
        # Encode image
        image_base64 = encode_image(image_path)
        
        # Import main module
        import main
        
        # Call run with image parameter
        result = main.run(image=image_base64)
        
        print(f"Prediction result: {json.dumps(result, indent=2)}")
        return result.get('status_code') == 200
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_requirements():
    """Check if all required files exist"""
    print("Checking requirements...")
    
    required_files = ['model.onnx', 'imagenet_classes.json']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} is missing")
            missing_files.append(file)
    
    if missing_files:
        print("\nMissing files detected. Creating dummy files for testing...")
        
        if 'imagenet_classes.json' in missing_files:
            # Create a minimal imagenet_classes.json
            classes = {str(i): f"class_{i}" for i in range(10)}
            classes.update({
                "281": "tabby cat",
                "285": "Egyptian cat",
                "291": "lion"
            })
            with open('imagenet_classes.json', 'w') as f:
                json.dump(classes, f, indent=2)
            print("Created imagenet_classes.json")
    
    return len(missing_files) == 0 or 'model.onnx' not in missing_files

if __name__ == "__main__":
    print("=" * 60)
    print("Cerebrium Model Test Suite")
    print("=" * 60)
    
    # Check requirements first
    if not check_requirements():
        print("\nCritical files missing. Please ensure model.onnx exists.")
        sys.exit(1)
    
    print("\n" + "=" * 60 + "\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--image" and len(sys.argv) > 2:
        # Test with specific image
        success = test_prediction(sys.argv[2])
    else:
        # Run health check
        success = test_health_check()
        
        # If health check passes, test with a sample image
        if success and len(sys.argv) > 1 and sys.argv[1] == "--run-tests":
            print("\n" + "=" * 60 + "\n")
            print("Running prediction test...")
            # Create a test image
            test_img = Image.new('RGB', (224, 224), color='blue')
            test_img.save('test_image.jpg')
            success = test_prediction('test_image.jpg')
    
    print("\n" + "=" * 60)
    print(f"Overall Test Result: {'PASSED' if success else 'FAILED'}")
    print("=" * 60)