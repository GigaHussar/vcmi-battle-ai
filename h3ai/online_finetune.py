"""online_finetune.py
----------------------
Quick fine‑tune on the battle that just ended.

Usage (inside `agent.py` **after** a battle finishes):

    from online_finetune import fine_tune_after_battle
    fine_tune_after_battle(game_id, epochs=3, lr=1e-4)

The function:
  • squeezes only the rows belonging to *game_id* out of MASTER_LOG.csv  
  • builds a tiny dataset of the chosen (state,action) pairs  
  • takes the current model weights, does a few gradient steps, saves them back

You can call it synchronously (training blocks gameplay)
or in a separate process / thread if you prefer.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from _paths_do_not_touch import MASTER_LOG, EXPORT_DIR, MODEL_WEIGHTS
from model import StateActionValueNet, FEATURE_DIM, STATE_DIM
import pandas as pd

# ---------------------------------------------------------------------
class BattleTurnDatasetOnline(torch.utils.data.Dataset):
    """Subset of the full log containing only one finished game."""
    def __init__(self, game_id: str,
                 log_csv: str = MASTER_LOG,
                 data_dir: str = EXPORT_DIR):
        self.df = (pd.read_csv(log_csv)
                     .query("game_id == @game_id")
                     .dropna(subset=["reward"])
                     .reset_index(drop=True))
        if self.df.empty:
            raise ValueError(f"No finished turns found for game_id={game_id}")
        self.base = Path(data_dir)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        g, prefix = str(row.game_id), row.state_prefix
        folder = Path(self.base)

        import numpy as np
        state = torch.from_numpy(
            np.concatenate([
                np.load(folder / f"battlefield_tensor_{prefix}.npy").astype("float32").flatten(),
                np.load(folder / f"creature_id_tensor_{prefix}.npy").astype("float32").flatten(),
                np.load(folder / f"faction_id_tensor_{prefix}.npy").astype("float32").flatten()
            ])
        )                                                   # (D,)

        acts = torch.from_numpy(
            np.load(folder / f"action_feats_{prefix}.npy").astype("float32")
        )                                                   # (K, F)
        chosen_idx = int((folder / f"chosen_idx_{prefix}.txt").read_text())
        chosen_feat = acts[chosen_idx]                      # (F,)

        reward = torch.tensor(float(row.reward), dtype=torch.float32)
        return {"state": state, "chosen_action": chosen_feat, "reward": reward}

# ---------------------------------------------------------------------
def _collate_online(batch):
    st  = torch.stack([b["state"] for b in batch])          # (B, D)
    act = torch.stack([b["chosen_action"] for b in batch])  # (B, F)
    rew = torch.tensor([b["reward"] for b in batch])
    return {"state": st, "action": act, "reward": rew}

# ---------------------------------------------------------------------
@torch.no_grad()
def _select_device(want: str | None = None):
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
def fine_tune_after_battle(game_id: int,
                           epochs: int = 3,
                           lr: float = 1e-4,
                           batch_size: int = 32,
                           device: str | None = None):
    """Do a few gradient steps on the just‑played battle."""
    device = _select_device(device)

    # ------------ dataset & loader ----------------------------------
    ds = BattleTurnDatasetOnline(game_id)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=_collate_online)

    # ------------ model & optimiser ---------------------------------
    net = StateActionValueNet().to(device)
    net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    for ep in range(epochs):
        for b in dl:
            st   = b["state"].to(device)
            act  = b["action"].to(device)
            targ = b["reward"].to(device)

            opt.zero_grad()
            pred = net(st, act)            # (B,)
            loss = crit(pred, targ)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

    # ------------ save ----------------------------------------------
    torch.save(net.state_dict(), MODEL_WEIGHTS)
    print(f"✓ online fine‑tune done on battle {game_id} (epochs={epochs})")

# ---------------------------------------------------------------------
if __name__ == "__main__":      # manual CLI usage
    import argparse, sys
    p = argparse.ArgumentParser(description="Online fine‑tune after one battle")
    p.add_argument("game_id", type=int)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", help="cpu | cuda | cuda:0 ...")
    a = p.parse_args()
    fine_tune_after_battle(a.game_id, a.epochs, a.lr, a.batch_size, a.device)
