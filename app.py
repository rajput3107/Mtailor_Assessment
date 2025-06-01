"""
Main Cerebrium application for image classification
"""
import base64
import io
import json
import numpy as np
from PIL import Image
from model import ClassificationModel
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model globally for reuse across requests
model = None

def initialize_model():
    """Initialize the model on cold start"""
    global model
    if model is None:
        logger.info("Initializing model...")
        start_time = time.time()
        model = ClassificationModel("/app/model.onnx")
        logger.info(f"Model initialized in {time.time() - start_time:.2f} seconds")
    return model

# Initialize model on import
initialize_model()

def predict(item, run_id, logger):
    """
    Main prediction endpoint for Cerebrium
    
    Args:
        item: Input data containing image
        run_id: Unique identifier for this run
        logger: Cerebrium logger instance
        
    Returns:
        Dictionary with prediction results
    """
    try:
        start_time = time.time()
        
        # Validate input
        if "image" not in item:
            return {
                "error": "No image provided in request",
                "status": "error"
            }
        
        # Decode image
        image_data = item["image"]
        
        # Handle base64 encoded image
        if isinstance(image_data, str):
            try:
                # Remove data URL prefix if present
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                return {
                    "error": f"Failed to decode image: {str(e)}",
                    "status": "error"
                }
        else:
            return {
                "error": "Image must be base64 encoded",
                "status": "error"
            }
        
        # Run prediction
        inference_start = time.time()
        predicted_class, confidence, probabilities = model.predict(image)
        inference_time = time.time() - inference_start
        
        # Get top 5 predictions
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_predictions = [
            {
                "class_id": int(idx),
                "confidence": float(probabilities[idx])
            }
            for idx in top5_indices
        ]
        
        total_time = time.time() - start_time
        
        response = {
            "status": "success",
            "predicted_class_id": predicted_class,
            "confidence": confidence,
            "top5_predictions": top5_predictions,
            "timing": {
                "inference_time": inference_time,
                "total_time": total_time
            },
            "run_id": run_id
        }
        
        logger.info(f"Prediction completed in {total_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "run_id": run_id
        }

# Health check endpoint
def health_check(item, run_id, logger):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "run_id": run_id
    }