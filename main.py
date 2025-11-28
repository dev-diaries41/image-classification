#! /usr/bin/env python3
import argparse
import os
import torch
import numpy as np
import random
from model import ImageClassifier, ImageClassifierWithMLP
from train import train, validate
from plot import plot_results
from inference import inference
from utils import load_class_names, get_new_filename

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)

def main():
    parser = argparse.ArgumentParser(description="Image classification pipeline")
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--class-names", type=str, required=True, help="Path to a text file with class names")
    common_parent.add_argument("-t", "--model-type", type=str, required=True, choices=["resnet", "mobilenet"],  help="Type of model to use")
    common_parent.add_argument("--use-hebb", action="store_true",  help="Use hebbian learning")
    subparsers = parser.add_subparsers(dest='mode')

    train_parser = subparsers.add_parser("train", parents=[common_parent])
    train_parser.add_argument("-d", "--data", type=str, required=True, help="Directory with training data")
    train_parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    infer_parser = subparsers.add_parser("infer", parents=[common_parent])
    infer_parser.add_argument("-m","--model", required=True, type=str, default="", help="Path to model to use for inference")
    infer_parser.add_argument("-i", "--input-image", required=True, type=str, help="Image file for inference")

    val_parser = subparsers.add_parser('validation')
    val_parser.add_argument("-d", "--data", required=True, type=str, help="Directory with training data")
    val_parser.add_argument('--ckpt', required=True, help='Path to saved checkpoint')
    
    args = parser.parse_args()

    class_names = load_class_names(args.class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "resnet":
        model = ImageClassifierWithMLP(num_classes=len(class_names)) if args.use_hebb else ImageClassifier(num_classes=len(class_names), architecture="resnet")
    elif args.model_type == "mobilenet":
        model = ImageClassifierWithMLP(num_classes=len(class_names), backbone="mobilenet") if args.use_hebb else ImageClassifier(num_classes=len(class_names), architecture="mobilenet")
    else:
        raise ValueError("The model_type provided is invalid. Choose between 'resnet' or 'mobilenet'")

    # TODO: Change classnames to dict
    if args.mode == "train":
        hebb_prefix = "hebb" if args.use_hebb else ""
        checkpoint_path = get_new_filename("checkpoints", f"checkpoint_best_{args.model_type}_{hebb_prefix}", ".pt")
        final_model_path = get_new_filename("models", f"model_{args.model_type}_{hebb_prefix}", ".pt")

        train_losses, test_losses, test_acc = train(
            model=model, 
            dataset_dir=args.data,
            class_names=class_names,
            device=device, checkpoint_path=checkpoint_path, 
            final_model_path=final_model_path, 
            epochs=args.epochs, lr=args.lr,
        )
        plot_path = get_new_filename("results", f"loss_accuracy_plot_{args.model_type}_{hebb_prefix}", ".png")
        plot_results(train_losses, test_losses, test_acc, output_path=plot_path)
    elif args.mode == "infer":
        print("Preparing inference...")
        print("Loading model...")
        model.load_state_dict(torch.load(args.model, map_location=device))
        sample_image = args.input_image
        prediction = inference(model, device, sample_image, class_names)
        print(f"Predicted class for the image '{sample_image}': {prediction}")
    elif args.mode == 'validation':
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
            loss, accuracy = validate(model, args.data, class_names)
            print(f" Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()