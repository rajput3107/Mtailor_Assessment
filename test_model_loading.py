# test_model_loading.py
import os
import sys

def test_model_loading():
    """Test if the model loads correctly"""
    print("Testing model loading...")
    
    # Check if model file exists
    model_path = "model.onnx"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return False
    
    print(f"✓ Model file exists: {model_path}")
    print(f"  Size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Try to load the model
    try:
        from model import ModelInference
        print("✓ Successfully imported ModelInference")
        
        # Try to create an instance
        model = ModelInference()
        print("✓ Successfully created ModelInference instance")
        
        # Check if model is loaded
        if hasattr(model, 'session') and model.session is not None:
            print("✓ ONNX model loaded successfully")
            return True
        else:
            print("✗ Model session not initialized")
            return False
            
    except Exception as e:
        print(f"✗ Error loading model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print("-" * 50)
    print(f"Model loading test {'PASSED' if success else 'FAILED'}")