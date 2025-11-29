#! /usr/bin/env python3
import argparse
import os
import torch
import numpy as np
import random
from model import ImageClassifier, ImageClassifierWithMLP
from train import train, validate, TrainConfig
from plot import plot_results
from inference import inference
from utils import get_new_filename

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
    subparsers = parser.add_subparsers(dest='mode')

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("-d", "--data", type=str, required=True, help="Directory with training data")
    train_parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("-p", "--patience", type=int, default=10, help="Max number of epochs without improvement before early stopping")
    train_parser.add_argument( "--lr-patience", type=int, default=3, help="LR scheduler patience")
    train_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    train_parser.add_argument("--use-hebb", action="store_true",  help="Use hebbian learning")
    train_parser.add_argument("-t", "--model-type", type=str, required=True, choices=["resnet", "mobilenet"],  help="Type of model to use")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("-m","--model", required=True, type=str, help="Path to model to use for inference")
    infer_parser.add_argument("-i", "--input-image", required=True, type=str, help="Image file for inference")

    val_parser = subparsers.add_parser('validation')
    val_parser.add_argument("-d", "--data", required=True, type=str, help="Directory with training data")
    val_parser.add_argument('--ckpt', required=True, help='Path to saved checkpoint')
    
    args = parser.parse_args()

    class_names = sorted(os.listdir(args.data))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        hebb_prefix = "hebb_" if args.use_hebb else ""
        checkpoint_path = get_new_filename("checkpoints", f"{args.model_type}_{hebb_prefix}", ".pt")
        final_model_path = get_new_filename("models", f"{args.model_type}_{hebb_prefix}", ".pt")

        config = TrainConfig(
            checkpoint_save_path=checkpoint_path, 
            model_save_path=final_model_path,
            model_type=args.model_type,
            use_hebb=args.use_hebb,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            lr_patience=args.lr_patience
            )
        
        if args.use_hebb:
            model = ImageClassifierWithMLP(num_classes=len(class_names), backbone=args.model_type)
        else:
            model =  ImageClassifier(num_classes=len(class_names), architecture=args.model_type)
       
        train_losses, test_losses, test_acc = train(model, config, args.data)
        plot_path = get_new_filename("results", f"loss_accuracy_plot_{args.model_type}_{hebb_prefix}", ".png")
        plot_results(train_losses, test_losses, test_acc, output_path=plot_path)
    elif args.mode == "infer":
        print("Running inference...")
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        config = TrainConfig(**ckpt['config'])

        if config.use_hebb:
            model = ImageClassifierWithMLP(num_classes=len(class_names), backbone=config.model_type)
        else:
            model =  ImageClassifier(num_classes=len(class_names), architecture=config.model_type)
        model.load_state_dict(ckpt['model_state'])
        sample_image = args.input_image
        prediction = inference(model, device, sample_image, class_names)
        print(f"Predicted class for the image '{sample_image}': {prediction}")
    elif args.mode == 'validation':
            print("Running validation...")
            ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
            config = TrainConfig(**ckpt['config'])

            if config.use_hebb:
                model = ImageClassifierWithMLP(num_classes=len(class_names), backbone=config.model_type)
            else:
                model = ImageClassifier(num_classes=len(class_names), architecture=config.model_type)
            
            model.load_state_dict(ckpt['model_state'])
            loss, accuracy = validate(model, args.data)
            print(f" Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()