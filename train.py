import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm

import os
from loader import ODIR_loader
from baseline_models import *

os.environ['TORCH_HOME'] = 'D:/pytorch_cache'

def train_and_evaluate(
    model, model_name, train_loader, val_loader, num_epochs=60, lr=1E-4, device="cuda:1"
):
    """
    Train a single model, saving results and metrics to its folder,
    including gender-specific AUC (male and female), and save the best model based on validation AUC.
    """
    # Create a directory for this model
    os.makedirs(model_name, exist_ok=True)

    device = torch.device(device)
    model.to(device)
    print(device)
    best_loss = 1E10  # Track the best validation AUC
    best_model_path = os.path.join(model_name, "best_model.pth")

    # Define loss and optimizer
    # class_weights = torch.tensor([4]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize history tracking
    train_loss_history, val_loss_history = [], []
    train_auc_history, val_auc_history = [], []
    train_male_auc_history, train_female_auc_history = [], []
    val_male_auc_history, val_female_auc_history = [], []

    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_loss, y_true_train, y_pred_train, male_train = 0, [], [], []

        for batch in train_loader:
            inputs, labels, males = (
                batch["input"].to(device),
                batch["label"].to(device),
                batch["male"],
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(dim=1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            male_train.extend(males.cpu().numpy())

        # Calculate overall and gender-specific AUC for training
        male_train = torch.tensor(male_train)
        train_auc = roc_auc_score(y_true_train, y_pred_train)
        train_male_auc = (
            roc_auc_score(
                [y for y, m in zip(y_true_train, male_train) if m == 1],
                [y for y, m in zip(y_pred_train, male_train) if m == 1],
            )
            if any(male_train == 1)
            else None
        )
        train_female_auc = (
            roc_auc_score(
                [y for y, m in zip(y_true_train, male_train) if m == 0],
                [y for y, m in zip(y_pred_train, male_train) if m == 0],
            )
            if any(male_train == 0)
            else None
        )

        # Append results
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        train_auc_history.append(train_auc)
        train_male_auc_history.append(train_male_auc)
        train_female_auc_history.append(train_female_auc)

        # Validation
        model.eval()
        val_loss, y_true_val, y_pred_val, male_val = 0, [], [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, males = (
                    batch["input"].to(device),
                    batch["label"].to(device),
                    batch["male"],
                )
                outputs = model(inputs)
                outputs = outputs.squeeze(dim=1)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                male_val.extend(males.cpu().numpy())

        # Calculate overall and gender-specific AUC for validation
        male_val = torch.tensor(male_val)
        val_auc = roc_auc_score(y_true_val, y_pred_val)
        val_male_auc = (
            roc_auc_score(
                [y for y, m in zip(y_true_val, male_val) if m == 1],
                [y for y, m in zip(y_pred_val, male_val) if m == 1],
            )
            if any(male_val == 1)
            else None
        )
        val_female_auc = (
            roc_auc_score(
                [y for y, m in zip(y_true_val, male_val) if m == 0],
                [y for y, m in zip(y_pred_val, male_val) if m == 0],
            )
            if any(male_val == 0)
            else None
        )

        # Append results
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc)
        val_male_auc_history.append(val_male_auc)
        val_female_auc_history.append(val_female_auc)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, "
            f"Train Male AUC: {train_male_auc}, Train Female AUC: {train_female_auc}, "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
            f"Val Male AUC: {val_male_auc}, Val Female AUC: {val_female_auc}"
        )

        # Save the best model if validation AUC improves
        if val_loss < best_loss:
            best_loss = val_loss
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save training and validation metrics to Excel
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train Loss": train_loss_history,
        "Train AUC": train_auc_history,
        "Train Male AUC": train_male_auc_history,
        "Train Female AUC": train_female_auc_history,
        "Validation Loss": val_loss_history,
        "Validation AUC": val_auc_history,
        "Validation Male AUC": val_male_auc_history,
        "Validation Female AUC": val_female_auc_history,
    })
    metrics_df.to_excel(os.path.join(model_name, "training_validation_metrics.xlsx"), index=False)

    return best_model_path, best_auc

def test_model(model, model_path, test_loader, device="cuda"):
    """
    Test the model using the best saved state and calculate AUC metrics.

    Args:
        model (torch.nn.Module): The model architecture.
        model_path (str): Path to the saved best model state.
        test_loader (DataLoader): DataLoader for testing data.
        device (str): Device to use ("cuda" or "cpu").
    """
    device = torch.device(device)
    model.to(device)

    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true_test, y_pred_test, male_test = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, males = (
                batch["input"].to(device),
                batch["label"].to(device),
                batch["male"],
            )
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            male_test.extend(males.cpu().numpy())

    # Overall and gender-specific AUC for testing
    male_test = torch.tensor(male_test)
    test_auc = roc_auc_score(y_true_test, y_pred_test)
    test_male_auc = (
        roc_auc_score(
            [y for y, m in zip(y_true_test, male_test) if m == 1],
            [y for y, m in zip(y_pred_test, male_test) if m == 1],
        )
        if any(male_test == 1)
        else None
    )
    test_female_auc = (
        roc_auc_score(
            [y for y, m in zip(y_true_test, male_test) if m == 0],
            [y for y, m in zip(y_pred_test, male_test) if m == 0],
        )
        if any(male_test == 0)
        else None
    )
    print(
        f"Test AUC: {test_auc:.4f}, Test Male AUC: {test_male_auc}, Test Female AUC: {test_female_auc}"
    )

    # Save testing metrics to Excel
    test_results = pd.DataFrame({
        "Metric": ["Overall AUC", "Male AUC", "Female AUC"],
        "Value": [test_auc, test_male_auc, test_female_auc],
    })
    test_results.to_excel(os.path.join(model_name, "test_results.xlsx"), index=False)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model_with_confusion_matrix(model, model_path, test_loader, device="cuda"):
    """
    Test the model using the best saved state and calculate AUC metrics along with confusion matrices.

    Args:
        model (torch.nn.Module): The model architecture.
        model_path (str): Path to the saved best model state.
        test_loader (DataLoader): DataLoader for testing data.
        device (str): Device to use ("cuda" or "cpu").
    """
    device = torch.device(device)
    model.to(device)

    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true_test, y_pred_test, male_test = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, males = (
                batch["input"].to(device),
                batch["label"].to(device),
                batch["male"],
            )
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            male_test.extend(males.cpu().numpy())

    # Convert predictions to binary labels using a threshold (default: 0.5)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_test]

    # Overall AUC
    test_auc = roc_auc_score(y_true_test, y_pred_test)
    print(f"Test AUC: {test_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true_test, y_pred_binary)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-DR", "DR"])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Overall)")
    plt.savefig(f"{model_path}_confusion_matrix_overall.png")
    plt.close()

    # Save confusion matrix as a DataFrame
    cm_df = pd.DataFrame(cm, index=["Non-DR", "DR"], columns=["Non-DR", "DR"])
    cm_df.to_excel(f"{model_path}_confusion_matrix_overall.xlsx")

    print("Confusion matrix saved as an image and Excel file.")



# Main script to train all models
if __name__ == "__main__":
    # Number of output classes
    num_classes = 2

    # Define the dataset and data loaders
    train_dataset = ODIR_loader(data_dir="./ODIR_Data/train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = ODIR_loader(data_dir="./ODIR_Data/val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataset = ODIR_loader(data_dir="./ODIR_Data/test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Train each model
    models_to_train = {
        # "VGG": get_vgg(num_classes),
        "ResNet": get_resnet(num_classes),
        "DenseNet": get_densenet(num_classes),
        # "EfficientNet": get_efficientnet(num_classes, model_name="efficientnet_b0"),
        # "ViT": get_vit(num_classes, model_name="vit_base_patch16_224"),
    }

    for model_name, model in models_to_train.items():
        print(f"Now {model_name}...")
        best_model_path, best_auc = train_and_evaluate(
            model, model_name, train_loader, val_loader, num_epochs=40, lr=1E-5, device="cuda"
        )
        print(f"Best Model Path: {best_model_path}, Best Validation AUC: {best_auc}")
        # best_model_path = f"{model_name}/best_model.pth"
        test_model_with_confusion_matrix(model, best_model_path, test_loader, device="cuda")