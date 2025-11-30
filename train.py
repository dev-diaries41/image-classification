import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from data import get_dataset
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


def train(model, config: TrainConfig, train_dataset_dir: str, val_dataset_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.lr_patience, factor=config.lr_factor, min_lr=config.min_lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_history = []
    val_history = []
    val_acc_history = []

    best_loss = float("inf")
    patience_counter = 0  # Tracks epochs without improvement
    use_hebb = hasattr(model, "mlp")

    train_dataset = get_dataset(train_dataset_dir)        
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = get_dataset(val_dataset_dir)        
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


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

        avg_val_loss, val_accuracy = evaluate(model, val_loader)
        val_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)
        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(avg_val_loss)

        # Early Stopping Check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
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
    return train_history, val_history, val_acc_history



def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
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


def test(model, dataset_dir):
    dataset = get_dataset(dataset_dir)    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return evaluate(model, loader)