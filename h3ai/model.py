import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import Dataset   
import pandas as pd
import numpy as np
import os

# Map your JSON type IDs → string labels (only 0,1,4,5 used)
ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
}

# Map your JSON type string labels to contiguous indices for embedding
ACTION_TYPE_LABELS = ["defend", "wait", "move", "melee"]
ACTION_LABEL_TO_IDX = {label: i for i, label in enumerate(ACTION_TYPE_LABELS)}  # {"defend":0, "wait":1, ...}
IDX_TO_ACTION_LABEL = {i: label for label, i in ACTION_LABEL_TO_IDX.items()}
N_ACTION_TYPES = len(ACTION_TYPE_LABELS)

# Board geometry for normalizing coords
WIDTH_FULL      = 17
WIDTH_PLAYABLE  = 15
HEIGHT          = 11
STATE_DIM       = 2970

def hex_to_coords(hex_id):
    """
    Convert a single hex ID into (x,y) on the full grid.
    """
    x = hex_id % WIDTH_FULL
    y = hex_id // WIDTH_FULL
    return x, y

def encode_hex_field(hex_idx):
    """
    Turn a hex index into [x_norm, y_norm, valid_flag].
    - If hex_idx < 0 or None, returns [0,0,0].
    - Otherwise returns [x/WIDTH_PLAYABLE, y/HEIGHT, 1].
    """
    if hex_idx is None or hex_idx < 0:
        return torch.tensor([0.0, 0.0, 0.0])
    x, y = hex_to_coords(hex_idx)
    # clamp to playable area before normalizing
    x = min(max(x, 0), WIDTH_PLAYABLE - 1)
    y = min(max(y, 0), HEIGHT        - 1)
    return torch.tensor([x / (WIDTH_PLAYABLE - 1),
                         y / (HEIGHT        - 1),
                         1.0])  # valid flag = 1

class ActionEncoder(nn.Module):
    """
    Turn each structured action dict into a fixed-length vector φ_a(a).
    - type_emb: converts action type → 8-dim embedding.
    - encode_hex_field: converts hex1/hex2 → 3 floats each.
    Final φ_a(a) size = 8 + 3 + 3 = 14.
    """
    def __init__(self, type_emb_dim: int = 8):
        super().__init__()
        # embedding table for the 4 action types
        self.type_emb = nn.Embedding(N_ACTION_TYPES, type_emb_dim)

    def forward(self, action_dicts: list[dict]) -> torch.Tensor:
        """
        action_dicts: list of length k, each dict has:
           - "type":    str in {"wait","defend","move","melee"}
           - "hex1":    int (or missing) for primary target
           - "hex2":    int (or missing) for secondary target (melee)
        Returns:
           Tensor of shape [k, 14], where each row is
             [ type_embed (8d) ∥ hex1_feat (3d) ∥ hex2_feat (3d) ].
        """
        batch = []
        for a in action_dicts:
            # 1) look up the integer ID of the action type
            t_idx = ACTION_LABEL_TO_IDX[a['type']]
            # 2) embed into a vector of size type_emb_dim
            e_t = self.type_emb(torch.tensor(t_idx, dtype=torch.long, device=self.type_emb.weight.device))

            # 3) encode first hex (move target or melee approach)
            f1 = encode_hex_field(a.get("hex1", -1))
            # 4) encode second hex (only for melee; else sentinel)
            f2 = encode_hex_field(a.get("hex2", -1))

            # 5) concatenate into one φ_a(a) vector of size 14
            batch.append(torch.cat([e_t, f1, f2], dim=0))

        # stack into tensor [k, 14]
        return torch.stack(batch, dim=0)

EMBED_DIM = 64   # joint embedding size

class BattleCommandScorer(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_feat_dim=14, embed_dim=EMBED_DIM):
        super().__init__()
        # 1) State projection: STATE_DIM → 256 → 128 → EMBED_DIM
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # 2) Raw action encoder (φₐ): list[dict] → [k, action_feat_dim]
        self.action_enc = ActionEncoder(type_emb_dim=8)

        # 3) Action projection: action_feat_dim → EMBED_DIM
        self.action_proj = nn.Sequential(
            nn.Linear(action_feat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, state_vec: torch.Tensor, action_dicts: list[dict]) -> torch.Tensor:
        """
        Args:
          state_vec:   Tensor [STATE_DIM]
          action_dicts: list of k action‐dicts from extract_all_possible_commands()
        Returns:
          logits:      Tensor [k] of **unnormalized** action-preferences
        """
        # Embed the state once
        s_emb = self.state_net(state_vec)           # [EMBED_DIM]

        # Encode & project actions
        a_feats = self.action_enc(action_dicts)      # [k, action_feat_dim]
        a_emb   = self.action_proj(a_feats)          # [k, EMBED_DIM]

        # Score = dot(s_emb, each a_emb[i])
        # expand s_emb to [k, EMBED_DIM]
        k      = a_emb.size(0)
        s_tile = s_emb.unsqueeze(0).expand(k, -1)    # [k, EMBED_DIM]
        scores = (s_tile * a_emb).sum(dim=1)         # [k]

        return scores

class BattleTurnDataset(Dataset):
    def __init__(self, log_csv_path: str, data_dir: str):
        # load and drop any incomplete episodes
        self.df = pd.read_csv(log_csv_path).dropna(subset=["reward"]).reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prefix = row.state_prefix               # e.g. "1624456789_5"
        # 1) load state tensors and flatten
        bf = np.load(os.path.join(self.data_dir, f"battlefield_tensor_{prefix}.npy"))
        cid = np.load(os.path.join(self.data_dir, f"creature_id_tensor_{prefix}.npy"))
        fid = np.load(os.path.join(self.data_dir, f"faction_id_tensor_{prefix}.npy"))
        state = np.concatenate([bf.flatten(), cid.flatten(), fid.flatten()]).astype(np.float32)
        # 2) load action-features matrix [k, F]
        actions = np.load(os.path.join(self.data_dir, f"action_feats_{prefix}.npy")).astype(np.float32)
        # 3) load chosen action index
        chosen_idx = int(open(os.path.join(self.data_dir, f"chosen_idx_{prefix}.txt")).read())
        # 4) load final reward
        reward = float(row.reward)
        # convert to torch
        return {
            "state":      torch.from_numpy(state),        # [S]
            "actions":    torch.from_numpy(actions),      # [k,F]
            "chosen_idx": torch.tensor(chosen_idx),       # []
            "reward":     torch.tensor(reward)            # []
        }