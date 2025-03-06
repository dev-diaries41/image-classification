import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

def evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0
    model.train()  # Switch back to training mode
    return avg_loss, accuracy

def train(model, train_loader, device, checkpoint_path, epochs=100, lr=0.001, test_loader=None, patience=10, lr_patience=3, factor=0.5, min_lr=1e-6):
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

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # Ensure model is in training mode
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                best_checkpoint_file = f"{checkpoint_path}_best.pt"
                torch.save(model.state_dict(), best_checkpoint_file)
                print(f"Best model saved: {best_checkpoint_file}")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered! Training stopped.")
                break

    filename = os.path.basename(checkpoint_path) 
    
    final_model = os.path.join("models", filename + ".pt")
    os.makedirs(os.path.dirname(final_model), exist_ok=True)   
    torch.save(model.state_dict(), final_model)
    print(f"Final model saved: {final_model}")

    return train_history, test_history, test_acc_history
