"""
Neural value function: V(state, action) → ℝ
------------------------------------------------
Training  : one chosen (state,action) pair → scalar value
Inference : (state, all candidate actions) → list of values

Shapes
------
state_vec      : (B, D)
chosen_actions : (B, F)          – during *training*
all_actions    : (B, K, F)       – during *play*
scores         : (B,) or (B, K)  – returned by forward()
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------
# Constants
WIDTH_FULL, WIDTH_PLAYABLE, HEIGHT = 17, 15, 11
STATE_DIM   = 2970       # ← 15×11×16 + 2×15×11
FEATURE_DIM = 14         # ← ActionEncoder output

ACTION_TYPE_LABELS = ["defend", "wait", "move", "melee"]
ACTION_LABEL_TO_IDX = {l: i for i, l in enumerate(ACTION_TYPE_LABELS)}

# --------------------------------------------------------------------
def hex_to_coords(hex_id: int):
    return hex_id % WIDTH_FULL, hex_id // WIDTH_FULL

def encode_hex_field(hex_idx):
    if hex_idx is None or hex_idx < 0:
        return torch.tensor([0.0, 0.0, 0.0])
    x, y = hex_to_coords(hex_idx)
    x = min(max(x, 0), WIDTH_PLAYABLE - 1)
    y = min(max(y, 0), HEIGHT - 1)
    return torch.tensor([x / (WIDTH_PLAYABLE - 1), y / (HEIGHT - 1), 1.0])

# --------------------------------------------------------------------
class ActionEncoder(nn.Module):
    """Turn list[dict] → (K, FEATURE_DIM) tensor."""
    def __init__(self, type_emb_dim: int = 8):
        super().__init__()
        self.type_emb = nn.Embedding(len(ACTION_TYPE_LABELS), type_emb_dim)

    def forward(self, actions: list[dict]) -> torch.Tensor:
        feats = []
        for a in actions:
            emb = self.type_emb(torch.tensor(ACTION_LABEL_TO_IDX[a["type"]]))
            f1 = encode_hex_field(a.get("hex1", -1))
            f2 = encode_hex_field(a.get("hex2", -1))
            feats.append(torch.cat([emb, f1, f2], dim=0))
        return torch.stack(feats, 0)  # (K, F)

# --------------------------------------------------------------------
class StateActionValueNet(nn.Module):
    """MLP estimating performance value."""
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = FEATURE_DIM,
                 hidden=(512, 256, 128)):
        super().__init__()
        dims = [state_dim + action_dim, *hidden, 1]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if d_out != 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, state_vec: torch.Tensor, action_feats: torch.Tensor):
        """Accepts (B, D) with either (B, F) or (B, K, F)."""
        if action_feats.dim() == 2:                 # training (chosen action)
            x = torch.cat([state_vec, action_feats], dim=1)   # (B, D+F)
            return self.mlp(x).squeeze(1)                     # (B,)
        # inference
        B, K, F = action_feats.shape
        state_rep = state_vec.unsqueeze(1).repeat(1, K, 1)    # (B, K, D)
        x = torch.cat([state_rep, action_feats], dim=2)       # (B, K, D+F)
        return self.mlp(x.view(B * K, -1)).view(B, K)         # (B, K)

# --------------------------------------------------------------------
class BattleTurnDataset(torch.utils.data.Dataset):
    """Loads tensors created by the logging helpers."""
    def __init__(self, log_csv: str, data_dir: str):
        import pandas as pd

        self.df = pd.read_csv(log_csv).dropna(subset=["reward"]).reset_index(drop=True)
        self.base = Path(data_dir)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        g, prefix = str(row.game_id), row.state_prefix
        folder = self.base / g

        # ----- state tensor -------------------------------------------------
        bf  = np.load(folder / f"battlefield_tensor_{prefix}.npy").astype(np.float32)
        cid = np.load(folder / f"creature_id_tensor_{prefix}.npy").astype(np.float32)
        fid = np.load(folder / f"faction_id_tensor_{prefix}.npy").astype(np.float32)
        state = torch.from_numpy(np.concatenate([bf.flatten(), cid.flatten(), fid.flatten()]))

        # ----- actions ------------------------------------------------------
        actions = torch.from_numpy(np.load(folder / f"action_feats_{prefix}.npy").astype(np.float32))
        chosen_idx = int((folder / f"chosen_idx_{prefix}.txt").read_text())
        chosen_feat = actions[chosen_idx]                         # (F,)

        reward = torch.tensor(float(row.reward), dtype=torch.float32)
        return {"state": state, "chosen_action": chosen_feat, "reward": reward}
