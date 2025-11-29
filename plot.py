# plot.py
import matplotlib.pyplot as plt
import os

def plot_results(train_losses, test_losses=None, test_accuracies=None, output_path="results.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    if test_losses:
        plt.plot(test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)
    
    if test_accuracies:
        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies, label="Test Accuracy", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved to {output_path}")
