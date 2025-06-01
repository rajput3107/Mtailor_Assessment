# main.py
import base64
import io
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self):
        self.session = ort.InferenceSession("model.onnx")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        try:
            with open('imagenet_classes.json', 'r') as f:
                labels_data = json.load(f)
                
            # Convert to dict if it's a list
            if isinstance(labels_data, list):
                self.class_labels = {str(i): label for i, label in enumerate(labels_data)}
                logger.info(f"Loaded {len(labels_data)} class labels from list format")
            else:
                self.class_labels = labels_data
                logger.info(f"Loaded {len(labels_data)} class labels from dict format")
        except Exception as e:
            logger.warning(f"Error loading labels: {e}. Using default labels.")
            self.class_labels = {str(i): f"class_{i}" for i in range(1000)}
    
    def preprocess_image(self, image):
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array
    
    def predict(self, image_data):
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        processed_image = self.preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: processed_image})
        logits = outputs[0]
        probabilities = self._softmax(logits[0])
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        predictions = []
        for idx in top_5_indices:
            # Now it will work with the dictionary format
            class_name = self.class_labels.get(str(idx), f"class_{idx}")
            predictions.append({
                "class": class_name,
                "confidence": float(probabilities[idx])
            })
        return predictions
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

try:
    model_instance = ModelInference()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model_instance = None

def run(image=None, run_id=None):
    if image is None:
        if model_instance is None:
            return {"result": {"status": "unhealthy", "error": "Model not initialized"}, "status_code": 503}
        try:
            test_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
            model_instance.session.run(None, {model_instance.input_name: test_input})
            return {"result": {"status": "healthy", "model": "loaded"}, "status_code": 200}
        except Exception as e:
            return {"result": {"status": "unhealthy", "error": str(e)}, "status_code": 503}
    
    try:
        if model_instance is None:
            return {"error": "Model not initialized", "status_code": 503}
        predictions = model_instance.predict(image)
        return {"predictions": predictions, "status_code": 200, "run_id": run_id}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e), "status_code": 500}

# Test immediately when run directly
if __name__ == "__main__":
    print("Testing main.py...")
    result = run()
    print(f"Health check: {json.dumps(result, indent=2)}")