import torch
import ai_edge_torch
import argparse
from model import ImageClassifier, ImageClassifierMobile

def convert_model(model_path, output_path, num_classes, model_type):
    print("Loading PyTorch model...")
    
    if model_type == "resnet":
        model = ImageClassifier(num_classes=num_classes)
    elif model_type == "mobilenet":
        model = ImageClassifierMobile(num_classes=num_classes)
    else:
        raise ValueError("Invalid model type. Choose between 'resnet' and 'mobilenet'.")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    print("Converting PyTorch model to TFLite...")
    sample_inputs = (torch.randn(1, 3, 224, 224),)
    
    edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
    edge_model.export(output_path)
    
    print(f"Model successfully converted to TFLite: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to TFLite format.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the PyTorch model file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the converted TFLite model.")
    parser.add_argument("-n", "--num_classes", type=int, required=True, help="Number of classes for the model.")
    parser.add_argument("-t", "--model_type", type=str, choices=["resnet", "mobilenet"], required=True, help="Type of model: 'resnet' or 'mobilenet'.")

    args = parser.parse_args()
    
    convert_model(args.model, args.output, args.num_classes, args.model_type)
