import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from DatasetAngle import SquatPhaseDataset
from collections import deque

# ====================== Configuration ======================
CONFIG = {
    'input_size': 5,  # Will be set automatically
    'hidden_size': 128,  # Increased hidden size
    'num_layers': 2,  # Increased LSTM layers
    'lr': 0.01,       # Higher initial learning rate
    'batch_size': 32,
    'epochs': 50,
    'threshold_pct': 50,  # Percentile for UP/DOWN threshold
    'sigma': 2.5,
    'early_stopping_patience': 10,
    'class_weights': torch.tensor([1.0, 1.0])  # Add class weights (e.g., for UP, DOWN, STABLE)
}

# ====================== Weighted Focal Loss ======================
class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        # Convert targets to long (integers) for CrossEntropyLoss
        targets = targets.long()
        
        # CrossEntropyLoss is used for multi-class classification
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(inputs, targets)
        
        # Compute the focal loss component
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ====================== Model Architecture ======================
class PhaseLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], 2)  # Output 3 classes (for UP, DOWN, STABLE)
        )
        
        # Initialize weights properly
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, x):
        x = x.float()
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])  # Output the last time step

# ====================== Feature Distribution Diagnostics ======================
def check_feature_distributions(dataset):
    """Analyze feature distributions by class"""
    features_up = []
    features_down = []
    features_stable = []
    
    for seq, label in dataset:
        if label == 0:  # UP
            features_up.append(seq.numpy())
        elif label == 1:  # DOWN
            features_down.append(seq.numpy())
    
    features_up = np.concatenate(features_up)
    features_down = np.concatenate(features_down)
    
    print("\n=== Feature Statistics ===")
    print(f"UP samples: {len(features_up)}")
    print(f"DOWN samples: {len(features_down)}")
    
    plt.figure(figsize=(15, 8))
    num_features = min(5, features_up.shape[1])  # Ensure we only use available features
    for i in range(num_features):
        plt.subplot(2, 3, i+1)
        plt.hist(features_up[:, i].ravel(), bins=50, alpha=0.5, label='UP')
        plt.hist(features_down[:, i].ravel(), bins=50, alpha=0.5, label='DOWN')
        plt.title(f'Feature {i} Distribution')
        plt.legend()
    plt.tight_layout()
    plt.show()

# ====================== Data Preparation ======================
def prepare_loaders():
    train_set = SquatPhaseDataset("Squat_Train.csv", 
                                seq_length=30,
                                threshold_pct=CONFIG['threshold_pct'],
                                sigma=CONFIG['sigma'])
    
    val_set = SquatPhaseDataset("Squat_Test.csv",
                              seq_length=30,
                              threshold_pct=CONFIG['threshold_pct'],
                              sigma=CONFIG['sigma'])

    # Run diagnostics
    print("=== Training Set Diagnostics ===")
    check_feature_distributions(train_set)
    
    CONFIG['input_size'] = train_set.data[0].shape[1]

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=train_set.collate_fn
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG['batch_size'],
        collate_fn=val_set.collate_fn
    )
    
    return train_loader, val_loader

# ====================== Training with Monitoring ======================
def train_model(train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhaseLSTM(CONFIG['input_size']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = WeightedFocalLoss(class_weights=CONFIG['class_weights'])  # Use Weighted Focal Loss
    
    # Tracking variables
    grad_norms = []
    train_losses = []  # Store avg train loss per epoch
    val_losses = []    # Store avg val loss per epoch
    best_f1 = 0
    patience_counter = 0
    lr_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        current_grad_norms = []
        
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient norm calculation
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            current_grad_norms.append(grad_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        grad_norms.extend(current_grad_norms)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.long().to(device)
                outputs = model(seq)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Avg Grad Norm: {np.mean(current_grad_norms):.4f}")
        print(f"  Confusion Matrix:\n{val_metrics['confusion_matrix']}")
        
        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), "models/lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            if (lr_counter >= 5):
                lr_counter = 0
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']:.2e}")
            else:
                lr_counter += 1
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(grad_norms)
    plt.title("Gradient Norms During Training")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model

# ====================== Evaluation ======================
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.to(device)
            outputs = model(seq).cpu()
            preds = torch.argmax(outputs, dim=1)  # Using argmax for multi-class classification
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute precision, recall, and F1 score for each class
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': (all_preds == all_labels).mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

# ====================== Main Execution ======================
if __name__ == "__main__":
    # Initialize and check data
    train_loader, val_loader = prepare_loaders()
    
    # Train model
    print("\n=== Starting Training ===")
    model = train_model(train_loader, val_loader)
    
    # Final evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_metrics = evaluate(model, val_loader, device)
    
    print("\n=== Final Evaluation ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(final_metrics['confusion_matrix'])
    print("\n=== Training Complete ===")