"""
Convert PyTorch model to ONNX format for efficient inference
"""
import torch
import torch.onnx
import numpy as np
from pytorch_model import Classifier, BasicBlock
import argparse
import os


def convert_pytorch_to_onnx(
    model_weights_path: str,
    output_path: str = "model.onnx",
    opset_version: int = 11
):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_weights_path: Path to PyTorch model weights (.pth file)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    # Initialize model - Use Classifier instead of PyTorchModel
    print("Loading PyTorch model...")
    model = Classifier(BasicBlock, [2, 2, 2, 2])  # ResNet18 architecture
    
    # Load weights
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
    
    model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    print("Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to {output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification successful!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument(
        '--weights',
        type=str,
        default='pytorch_model_weights.pth',
        help='Path to PyTorch model weights'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model.onnx',
        help='Output ONNX model path'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version'
    )
    
    args = parser.parse_args()
    convert_pytorch_to_onnx(args.weights, args.output, args.opset)


if __name__ == "__main__":
    main()