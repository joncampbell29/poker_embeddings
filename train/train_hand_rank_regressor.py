import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch_geometric as tg
import torch
import torch.optim as optim
import argparse
import os
import torch.nn.functional as F
from poker_embeddings.models.card import CardGNNRegressor
from poker_embeddings.poker_utils.datasets import UCIrvineDataset
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(model, trainloader, optimizer, scheduler=None, device=None,
                valloader=None, epochs=50,
                leftoff=0, save=False, save_dir="./model_weights/hand_rank_predictor", save_interval=None):

    train_losses = []
    val_losses = []
    try:
        for epoch in range(epochs):
            tot_train_loss = 0

            model.train()
            for batch_data in trainloader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()

                output = model(batch_data)

                batch_loss = F.mse_loss(output, batch_data.y.float().squeeze(-1))
                batch_loss.backward()
                optimizer.step()
                tot_train_loss += batch_loss.item()

            avg_train_loss = tot_train_loss / len(trainloader)
            train_losses.append(avg_train_loss)

            if valloader is not None:
                model.eval()
                tot_val_loss = 0

                with torch.no_grad():
                    for batch_data in valloader:
                        batch_data = batch_data.to(device)
                        logits = model(batch_data)
                        batch_loss = F.mse_loss(logits, batch_data.y.float().squeeze(-1))
                        tot_val_loss += batch_loss.item()

                avg_val_loss = tot_val_loss / len(valloader)
                val_losses.append(avg_val_loss)

            if valloader is not None:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            if save:
                if (epoch + 1) % save_interval == 0:
                    torch.save(model.state_dict(), os.path.join(save_dir, f"hand_rank_predictor{leftoff+epoch+1}.pth"))
                if epoch+1 == epochs:
                    torch.save(model.state_dict(), os.path.join(save_dir, f"hand_rank_predictor{leftoff+epoch+1}.pth"))

            if scheduler is not None:
                    scheduler.step()
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(save_dir, f"hand_rank_predictor_intrp{leftoff+epoch+1}.pth"))
        print(f"Ending at Epoch {epoch+1}/{epochs}.")
    finally:
        if valloader is not None:
            return {"train_loss":train_losses,
                    "val_loss":val_losses}
        else:
            return {'train_loss':train_losses}


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg["model"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["results"]["save_dir"], exist_ok=True)

    X = pd.read_csv(cfg["data"]["X_path"])
    y = pd.read_csv(cfg["data"]["y_path"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=29, stratify=y['CLASS']
        )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    train_dataset = UCIrvineDataset(X_train, y_train, add_random_cards=True, use_card_ids=True, graph=True, normalize_x=True)
    val_dataset = UCIrvineDataset(X_val, y_val, add_random_cards=True, use_card_ids=True, graph=True, normalize_x=True)

    trainloader = tg.loader.DataLoader(
        train_dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        persistent_workers=True
    )
    valloader = tg.loader.DataLoader(
        val_dataset,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"]["pin_memory"],
        persistent_workers=True
    )

    if cfg["training"]["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["training"]["device"])

    model_param_cfg = cfg["model"]["params"]
    model = CardGNNRegressor(**model_param_cfg).to(device)
    if cfg["model"]["weights_path"]:
        model.load_state_dict(torch.load(cfg["model"]["weights_path"], weights_only=True))

    opt_cfg = cfg["training"]["optimizer"]
    optimizer_class = getattr(optim, opt_cfg["type"])
    optimizer = optimizer_class(
        model.parameters(),
        **{p_name: p_val for p_name, p_val in opt_cfg.items() if p_name != "type"})

    scheduler = None
    sched_cfg = cfg["training"]["scheduler"]
    if sched_cfg.get("use_scheduler", False):
        scheduler_class = getattr(optim.lr_scheduler, sched_cfg["type"])
        scheduler = scheduler_class(
            optimizer,
            **{p_name: p_val for p_name, p_val in sched_cfg.items() if p_name not in ["type", "use_scheduler"]})

    res = train_model(
        model=model,
        trainloader=trainloader,
        valloader=valloader if cfg['training']['val_during_training'] else None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=cfg["training"]["epochs"],
        leftoff=cfg["training"]["start_epoch"],
        save=cfg["training"]["save_weights"],
        save_dir=cfg["model"]["save_dir"],
        save_interval=cfg["training"]["save_interval"]
    )
    min_len = min(len(v) for v in res.values())
    for k in res.keys():
        res[k] = res[k][:min_len]
    res_df = pd.DataFrame.from_dict(res)

    epochs = res_df.shape[0]
    res_df["epochs"] = np.arange(cfg["training"]["start_epoch"] + 1, cfg["training"]["start_epoch"] + 1 + epochs)
    res_df.to_csv(
        os.path.join(
            cfg["results"]["save_dir"], f"hand_rank_predictor_train_loss_{cfg['training']['start_epoch'] + epochs}.csv"),
            index=False
            )

    with open(os.path.join(cfg["results"]["save_dir"], f"config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

