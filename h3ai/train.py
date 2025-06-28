import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import StateActionValueNet, BattleTurnDataset
from _paths_do_not_touch import MASTER_LOG, EXPORT_DIR, MODEL_WEIGHTS

# --------------------------------------------------------------------
def collate(batch):
    st   = torch.stack([b["state"] for b in batch])          # (B, D)
    act  = torch.stack([b["chosen_action"] for b in batch])  # (B, F)
    rew  = torch.tensor([b["reward"] for b in batch])
    return {"state": st, "action": act, "reward": rew}

# --------------------------------------------------------------------
def train(epochs=10, batch_size=32, lr=1e-3, val_split=0.2, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = BattleTurnDataset(str(MASTER_LOG), str(EXPORT_DIR))
    n_val = int(len(ds) * val_split)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = StateActionValueNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()
    best  = float("inf")

    for ep in range(1, epochs + 1):
        # ------------- train ----------------------------------------------
        model.train(); run = 0.0
        for b in dl_train:
            st = b["state"].to(device)
            act = b["action"].to(device)
            tgt = b["reward"].to(device)

            opt.zero_grad()
            pred = model(st, act)                 # (B,)
            loss = crit(pred, tgt)
            loss.backward()
            opt.step()
            run += loss.item() * st.size(0)
        tr_loss = run / len(train_ds)

        # ------------- validation -----------------------------------------
        model.eval(); run = 0.0
        with torch.no_grad():
            for b in dl_val:
                st = b["state"].to(device)
                act = b["action"].to(device)
                tgt = b["reward"].to(device)
                pred = model(st, act)
                run += crit(pred, tgt).item() * st.size(0)
        val_loss = run / len(val_ds)
        print(f"Epoch {ep}/{epochs} – train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_WEIGHTS)
            print(f"  ✓ saved best to {MODEL_WEIGHTS}")

# --------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--device")
    a = p.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.val_split, a.device)
