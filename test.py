"""
Comprehensive test suite for the classification model
"""
import unittest
import numpy as np
import os
import tempfile
from PIL import Image
import torch
from model import ImagePreprocessor, ONNXModel, ClassificationModel
from convert_to_onnx import convert_pytorch_to_onnx
from pytorch_model import Classifier, BasicBlock


class TestImagePreprocessor(unittest.TestCase):
    """Test image preprocessing functionality"""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        
    def test_preprocess_shape(self):
        """Test output shape of preprocessed image"""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        processed = self.preprocessor.preprocess(dummy_image)
        
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        self.assertEqual(processed.dtype, np.float32)
    
    def test_preprocess_grayscale(self):
        """Test preprocessing of grayscale images"""
        grayscale_image = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
        processed = self.preprocessor.preprocess(grayscale_image)
        
        self.assertEqual(processed.shape, (1, 3, 224, 224))
    
    def test_preprocess_normalization(self):
        """Test that normalization is applied correctly"""
        # Create white image
        white_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
        processed = self.preprocessor.preprocess(white_image)
        
        # Check that values are normalized
        self.assertTrue(np.all(processed > 0))  # After normalization
        self.assertTrue(np.all(processed < 5))  # Reasonable upper bound
    
    def test_preprocess_pil_image(self):
        """Test preprocessing of PIL images"""
        pil_image = Image.new('RGB', (300, 400), color='red')
        processed = self.preprocessor.preprocess(pil_image)
        
        self.assertEqual(processed.shape, (1, 3, 224, 224))


class TestONNXConversion(unittest.TestCase):
    """Test PyTorch to ONNX conversion"""
    
    def test_conversion(self):
        """Test model conversion process"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy weights - using Classifier model
            model = Classifier(BasicBlock, [2, 2, 2, 2])
            weights_path = os.path.join(tmpdir, 'dummy_weights.pth')
            torch.save(model.state_dict(), weights_path)
            
            # Convert to ONNX
            onnx_path = os.path.join(tmpdir, 'test_model.onnx')
            result_path = convert_pytorch_to_onnx(weights_path, onnx_path)
            
            self.assertTrue(os.path.exists(result_path))
            self.assertEqual(result_path, onnx_path)


class TestClassificationModel(unittest.TestCase):
    """Test the complete classification pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Create test ONNX model"""
        cls.test_dir = tempfile.mkdtemp()
        
        # Create dummy model - using Classifier model
        model = Classifier(BasicBlock, [2, 2, 2, 2])
        weights_path = os.path.join(cls.test_dir, 'test_weights.pth')
        torch.save(model.state_dict(), weights_path)
        
        # Convert to ONNX
        cls.onnx_path = os.path.join(cls.test_dir, 'test_model.onnx')
        convert_pytorch_to_onnx(weights_path, cls.onnx_path)
    
    def setUp(self):
        self.model = ClassificationModel(self.onnx_path)
    
    def test_predict_output_format(self):  # Fixed indentation here
        """Test prediction output format"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        class_id, confidence, probabilities = self.model.predict(dummy_image)
        
        # Check output types
        self.assertIsInstance(class_id, int)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(probabilities, np.ndarray)
        
        # Check ranges
        self.assertTrue(0 <= class_id < 1000)
        self.assertTrue(0 <= confidence <= 1)
        self.assertEqual(len(probabilities), 1000)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=5)
    
    def test_known_images(self):
        """Test with known images if available"""
        test_images = [
            ("test_images/n01440764_tench.JPEG", 0),
            ("test_images/n01667114_mud_turtle.JPEG", 35)
        ]
        
        for image_path, expected_class in test_images:
            if os.path.exists(image_path):
                class_id, confidence, _ = self.model.predict(image_path)
                print(f"Image: {image_path}, Predicted: {class_id}, Expected: {expected_class}")
                # Note: With random weights, we won't get correct predictions
                # This test just ensures the pipeline works
                self.assertTrue(0 <= class_id < 1000)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        shutil.rmtree(cls.test_dir)


class TestModelPerformance(unittest.TestCase):
    """Test model performance requirements"""
    
    def setUp(self):
        # Create test model
        self.test_dir = tempfile.mkdtemp()
        model = Classifier(BasicBlock, [2, 2, 2, 2])  # Using Classifier model
        weights_path = os.path.join(self.test_dir, 'test_weights.pth')
        torch.save(model.state_dict(), weights_path)
        onnx_path = os.path.join(self.test_dir, 'test_model.onnx')
        convert_pytorch_to_onnx(weights_path, onnx_path)
        self.model = ClassificationModel(onnx_path)
    
    def test_inference_speed(self):
        """Test that inference meets speed requirements"""
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Warm up
        _ = self.model.predict(dummy_image)
        
        # Time inference
        import time
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = self.model.predict(dummy_image)
        
        avg_time = (time.time() - start_time) / num_runs
        
        print(f"Average inference time: {avg_time:.3f}s")
        # Should be well under 2-3 seconds requirement
        self.assertLess(avg_time, 2.0)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)


def run_all_tests():
    """Run all tests and generate report"""
    print("Running Classification Model Test Suite")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImagePreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestONNXConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationModel))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()