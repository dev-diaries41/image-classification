import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import read_dataset_dir
from data import ImageDataset

def evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    use_hebb = hasattr(model, "mlp")

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
    model.train()  # Switch back to training mode
    return avg_loss, accuracy

def train(model, dataset_dir, class_names, device, checkpoint_path, final_model_path, epochs=100, lr=0.001, patience=10, lr_patience=3, factor=0.5, min_lr=1e-6, batch_size = 8):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Learning rate scheduler: Reduce LR if test loss doesn’t improve
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=lr_patience, factor=factor, min_lr=min_lr, verbose=True)

    train_history = []
    test_history = []
    test_acc_history = []

    best_loss = float("inf")
    patience_counter = 0  # Tracks epochs without improvement
    use_hebb = hasattr(model, "mlp")

    file_paths, labels = read_dataset_dir(dataset_dir)
    numbered_labels = [class_names.index(label) for label in labels]
    dataset = ImageDataset(file_paths, numbered_labels)        
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # Ensure model is in training mode
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if use_hebb:
                y_pred, acts = model(images, return_activations = True)
            else:
                 y_pred = model(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            
            if use_hebb and (epoch % 2 == 0):
                    model.mlp.apply_hebb(acts, y_true = labels, y_pred = y_pred, gate_threshold = 0.5)

            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_history.append(avg_train_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Evaluate on test set if provided
        if test_loader:
            avg_test_loss, test_accuracy = evaluate(model, test_loader, device, criterion)
            test_history.append(avg_test_loss)
            test_acc_history.append(test_accuracy)
            print(f"Epoch {epoch+1} Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            # Adjust learning rate if test loss plateaus
            scheduler.step(avg_test_loss)

            # Early Stopping Check
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                patience_counter = 0  # Reset patience
                # Save the best model
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved: {checkpoint_path}")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered! Training stopped.")
                break
    
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    return train_history, test_history, test_acc_history
