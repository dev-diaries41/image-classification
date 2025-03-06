#! /usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, random_split
from model import ImageClassifier, ImageClassifierMobile
from data import Dataset
from train import train
from plot import plot_results
from inference import inference
from utils import read_dataset_dir, get_labels, load_class_names
import argparse

def main():
    parser = argparse.ArgumentParser(description="Image classification pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True,
                        help="Choose 'train' to train the model or 'inference' to run enhancement on an image file")
    parser.add_argument("--dataset", type=str, help="Directory with training data")
    parser.add_argument("--input", type=str, help="Image file for inference")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/screenshot_model", 
                        help="Filename to save/load model checkpoint")
    parser.add_argument("--model", type=str, default="", help="Path to model to use for inference")
    parser.add_argument("--plot_file", type=str, default="results/loss_accuracy_plot.png", help="Filename for saving the loss plot")
    parser.add_argument("--class_names", type=str, required=True, help="Path to a text file with class names")
    args = parser.parse_args()

    class_names = load_class_names(args.class_names)

    if args.mode == "train":
        if not args.dataset:
            raise ValueError("For training, please provide dataset with --dataset")
        
    if args.mode == "inference":
        if not args.model or not args.input:
            raise ValueError("For inference, please provide model path with --model and input image with --input")
        if not os.path.exists(args.input):
            raise ValueError("The input image you provided does not exist.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImageClassifierMobile(num_classes=len(class_names))
    
    if os.path.exists(args.model):
        print("Loading existing model...")
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        print("No existing model found. Using default pre-trained model (will train if dataset is provided).")

    if args.mode == "train":
        print("Training model...")
        image_paths = read_dataset_dir(args.dataset)
        labels = get_labels(image_paths)
        print(f"Dataset size: ", len(labels))
        dataset = Dataset(image_paths, labels)
        
        train_size = int(0.75 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        train_losses, test_losses, test_acc = train(
            model, train_loader, device, checkpoint_path=args.checkpoint, epochs=args.epochs, lr=args.lr,
              test_loader=test_loader
        )
        plot_results(train_losses, test_losses, test_acc, output_path=args.plot_file)
    elif args.mode == "inference":
        print("Preparing inference...")
        sample_image = args.input
        prediction = inference(model, device, sample_image, class_names)
        print(f"Predicted class for the image '{sample_image}': {prediction}")

if __name__ == "__main__":
    main()