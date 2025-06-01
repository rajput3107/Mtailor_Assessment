"""
ONNX model inference and image preprocessing utilities
"""
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Union, Tuple, List
import cv2


class ImagePreprocessor:
    """Handle image preprocessing for the model"""
    
    def __init__(self):
        self.input_size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        
        # Ensure image is in RGB format (not BGR)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to 224x224 using bilinear interpolation
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize using ImageNet statistics
        image = (image - self.mean) / self.std
        
        # Transpose to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)


class ONNXModel:
    """ONNX model loading and inference"""
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX model
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def _load_model(self) -> ort.InferenceSession:
        """Load ONNX model with appropriate providers"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            session = ort.InferenceSession(self.model_path, providers=providers)
            print(f"Model loaded with providers: {session.get_providers()}")
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on input tensor
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Model output probabilities
        """
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor}
            )
            return outputs[0]
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")


class ClassificationModel:
    """Combined model for image classification"""
    
    def __init__(self, model_path: str):
        self.preprocessor = ImagePreprocessor()
        self.model = ONNXModel(model_path)
        self.num_classes = 1000  # ImageNet classes
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Tuple[int, float, np.ndarray]:
        """
        Predict class for input image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (predicted_class_id, confidence, all_probabilities)
        """
        # Preprocess image
        input_tensor = self.preprocessor.preprocess(image)
        
        # Run inference
        outputs = self.model.predict(input_tensor)
        
        # Get probabilities (apply softmax if needed)
        probabilities = self._softmax(outputs[0])
        
        # Get predicted class
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence, probabilities
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()