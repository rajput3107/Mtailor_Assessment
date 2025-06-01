# simple_test.py
import subprocess
import sys
import json

def test_main_directly():
    """Test main.py by running it as a subprocess"""
    
    # Create a test runner script
    test_script = '''
import sys
sys.path.insert(0, '.')

# Execute main.py content
with open('main.py', 'r') as f:
    exec(f.read(), globals())

# Now test the run function
print("Testing health check...")
result = run()
print(f"Health check result: {json.dumps(result, indent=2)}")

# Test with image
import base64
from PIL import Image

print("\\nCreating test image...")
test_img = Image.new('RGB', (224, 224), color='red')
test_img.save('test_image_temp.jpg')

with open('test_image_temp.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

print("Testing with image...")
result = run(image=image_base64)
print(f"Prediction result: {json.dumps(result, indent=2)}")
'''
    
    # Write the test script
    with open('temp_test_runner.py', 'w') as f:
        f.write(test_script)
    
    # Run it
    try:
        result = subprocess.run([sys.executable, 'temp_test_runner.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    finally:
        # Cleanup
        import os
        if os.path.exists('temp_test_runner.py'):
            os.remove('temp_test_runner.py')
        if os.path.exists('test_image_temp.jpg'):
            os.remove('test_image_temp.jpg')

if __name__ == "__main__":
    print("=" * 60)
    print("Direct Main.py Test")
    print("=" * 60)
    
    success = test_main_directly()
    
    print("\n" + "=" * 60)
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print("=" * 60)