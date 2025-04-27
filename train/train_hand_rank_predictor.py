import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch_geometric as tg
import torch
import argparse
import os
import torch.nn.functional as F
from poker_embeddings.models.card import CardGNN
from poker_embeddings.poker_utils.constants import DECK_DICT
from poker_embeddings.poker_utils.datasets import UCIrvineDataset
from sklearn.metrics import classification_report, confusion_matrix

def train_model(model, trainloader, optimizer, scheduler=None, device=None,
                valloader=None, epochs=50, leftoff=0, save=True):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    class_weights = torch.load("./model_weights/class_weights.pt", weights_only=True).to(device)
    for epoch in range(epochs):
        tot_train_loss = 0
        correct_train = 0
        total_train = 0

        model.train()
        for batch_data in trainloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            logits = model(batch_data)

            batch_loss = F.cross_entropy(logits, batch_data.y, weight=class_weights)
            batch_loss.backward()
            optimizer.step()

            tot_train_loss += batch_loss.item()
            preds = logits.argmax(dim=1)
            correct_train += (preds == batch_data.y).sum().item()
            total_train += batch_data.y.size(0)

        avg_train_loss = tot_train_loss / len(trainloader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        if valloader is not None:
            model.eval()
            tot_val_loss = 0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for batch_data in valloader:
                    batch_data = batch_data.to(device)
                    logits = model(batch_data)
                    batch_loss = F.cross_entropy(logits, batch_data.y, weight=class_weights)

                    tot_val_loss += batch_loss.item()
                    preds = logits.argmax(dim=1)
                    correct_val += (preds == batch_data.y).sum().item()
                    total_val += batch_data.y.size(0)

            avg_val_loss = tot_val_loss / len(valloader)
            val_acc = correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

        if valloader is not None:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if save:
            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), f"../model_weights/hand_strength_predictor{leftoff+epoch+1}.pth")

        if scheduler is not None:
                scheduler.step()

    if valloader is not None:
        return {"train_loss":train_losses,
                "val_loss":val_losses,
                "train_accuracy":train_accuracies,
                "val_accuracy":val_accuracies}
    else:
        return {'train_loss':train_losses, "train_accuracy":train_accuracies}

def get_classification_report(model, dataloader, device=None):
    all_preds = []
    all_labels = []
    all_indices = []
    class_names = [
        "nothing", "one_pair", "two_pair", "three_of_a_kind", "straight",
        "flush", "full_house", "four_of_a_kind", "straight_flush", "royal_flush"
        ]
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)

            logits = model(batch_data)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch_data.y.cpu())
            all_indices.extend(range(i * dataloader.batch_size, (i + 1) * dataloader.batch_size))
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return {'report':report, 'confusion_matrix':cm,'labels':all_labels,'pred':all_preds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hand rank predictor model.")
    parser.add_argument("--test_size", type=float, default=0.6, help="Percentage of Data to use for test set")
    parser.add_argument("--weights", type=str, default="", help="Path to the model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--leftoff", type=int, default=0, help="Epoch to start training from")
    parser.add_argument("--save_weights", type=bool, default=True, help="Save model weights")
    args = parser.parse_args()


    X = pd.read_csv("./data/uc_irvine/X.csv")
    y = pd.read_csv("./data/uc_irvine/y.csv")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=29, stratify=y['CLASS']
        )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    train_dataset = UCIrvineDataset(X_train, y_train, add_random_cards=True, use_card_ids=True,
                           graph=True, normalize_x=True)
    val_dataset = UCIrvineDataset(X_val, y_val, add_random_cards=True, use_card_ids=True,
                           graph=True, normalize_x=True)

    trainloader = tg.loader.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=10,
        pin_memory=True
        )
    valloader = tg.loader.DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=10,
        pin_memory=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CardGNN().to(device)
    if args.weights != '':
        args.weights = os.path.join("./model_weights", args.weights)
        model.load_state_dict(torch.load(args.weights, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    res = train_model(
        model=model,
        trainloader=trainloader,
        valloader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        leftoff=args.leftoff,
        save=args.save_weights
        )
    res = pd.DataFrame.from_dict(res)
    res['epochs'] = np.arange(args.leftoff+1, args.leftoff+1 + args.epochs)
    res.to_csv(f"./results/hand_rank_predictor_train_loss{args.leftoff+args.epochs}.csv", index=False)
    pred_res = get_classification_report(model, valloader, device=device)
    classification_report = pd.DataFrame.from_dict(pred_res['report']).T
    classification_report.to_csv(f"./results/hand_rank_predictor_classification_report{args.leftoff+args.epochs}.csv")

    class_names = [
        "High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"
        ]
    cm = pd.DataFrame(pred_res['confusion_matrix'],
                      index=class_names, columns=class_names)

    cm.to_csv(f"./results/hand_rank_predictor_confusion_matrix{args.leftoff+args.epochs}.csv")

