import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import read_dataset_dir
from data import ImageDataset
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class TrainConfig:
    checkpoint_save_path: str
    model_save_path: str
    model_type: str
    use_hebb: bool
    epochs: int = 100
    batch_size: int = 8
    lr: float = 0.0001
    patience: int = 20
    lr_patience: int = 3
    lr_factor: float=0.5
    min_lr: float=1e-6


def train(model, config: TrainConfig, dataset_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.lr_patience, factor=config.lr_factor, min_lr=config.min_lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_history = []
    test_history = []
    test_acc_history = []

    best_loss = float("inf")
    patience_counter = 0  # Tracks epochs without improvement
    use_hebb = hasattr(model, "mlp")

    file_paths, labels, class_names = read_dataset_dir(dataset_dir)
    numbered_labels = [class_names.index(label) for label in labels]
    dataset = ImageDataset(file_paths, numbered_labels)        
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for epoch in range(config.epochs):
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if use_hebb:
                y_pred, acts = model(images, return_activations = True)
            else:
                 y_pred = model(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            
            if use_hebb and epoch > 1:
                model.mlp.apply_hebb(acts, loss = loss.item(), avg_loss = np.mean(train_history[-5:] if len(train_history) >= 1 else 1))

            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_history.append(avg_train_loss)

        avg_test_loss, test_accuracy = evaluate(model, test_loader)
        test_history.append(avg_test_loss)
        test_acc_history.append(test_accuracy)
        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        scheduler.step(avg_test_loss)

        # Early Stopping Check
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0 
            torch.save({"model_state": model.state_dict(), "config": asdict(config)}, config.checkpoint_save_path)
            print(f"Best model saved: {config.checkpoint_save_path}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered! Training stopped.")
            break
    
    torch.save({"model_state": model.state_dict(), "config": asdict(config)}, config.model_save_path)
    print(f"Final model saved: {config.model_save_path}")
    return train_history, test_history, test_acc_history



def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    use_hebb = hasattr(model, "mlp")
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if use_hebb:
                y_pred, _ = model(images, return_activations = False)
            else:
                 y_pred = model(images)
            loss = criterion(y_pred, labels)
            total_loss += loss.item()
            preds = y_pred.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    model.train()
    return avg_loss, accuracy


def validate(model, dataset_dir):
    file_paths, labels, class_names = read_dataset_dir(dataset_dir)
    numbered_labels = [class_names.index(label) for label in labels]
    dataset = ImageDataset(file_paths, numbered_labels)        
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return evaluate(model, loader)

