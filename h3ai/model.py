import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

# Action encoding utilities — used by file2.py to precompute features
# Map your JSON type IDs → string labels
ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
}
# Map string labels to contiguous indices for embedding
ACTION_TYPE_LABELS = ["defend", "wait", "move", "melee"]
ACTION_LABEL_TO_IDX = {label: i for i, label in enumerate(ACTION_TYPE_LABELS)}

# Board geometry for normalizing coords
WIDTH_FULL      = 17
WIDTH_PLAYABLE  = 15
HEIGHT          = 11
STATE_DIM       = 2970

EMBED_DIM = 64   # joint embedding size
FEATURE_DIM = 14 # precomputed action feature dimension


def hex_to_coords(hex_id):
    x = hex_id % WIDTH_FULL
    y = hex_id // WIDTH_FULL
    return x, y


def encode_hex_field(hex_idx):
    """
    Turn a hex index into [x_norm, y_norm, valid_flag].
    """
    if hex_idx is None or hex_idx < 0:
        return torch.tensor([0.0, 0.0, 0.0])
    x, y = hex_to_coords(hex_idx)
    x = min(max(x, 0), WIDTH_PLAYABLE - 1)
    y = min(max(y, 0), HEIGHT        - 1)
    return torch.tensor([x / (WIDTH_PLAYABLE - 1), y / (HEIGHT - 1), 1.0])


class ActionEncoder(nn.Module):
    """
    Converts list[dict] of raw action descriptions into 14-dim vectors.
    Used by file2.py to precompute and save .npy files.
    """
    def __init__(self, type_emb_dim: int = 8):
        super().__init__()
        self.type_emb = nn.Embedding(len(ACTION_TYPE_LABELS), type_emb_dim)

    def forward(self, action_dicts: list[dict]) -> torch.Tensor:
        batch = []
        for a in action_dicts:
            t_idx = ACTION_LABEL_TO_IDX[a['type']]
            e_t   = self.type_emb(torch.tensor(t_idx, device=self.type_emb.weight.device))
            f1    = encode_hex_field(a.get("hex1", -1))
            f2    = encode_hex_field(a.get("hex2", -1))
            batch.append(torch.cat([e_t, f1, f2], dim=0))
        return torch.stack(batch, dim=0)


# ------------------------------------------------------------------
# Model consumes precomputed action features, so does not re-encode
# ------------------------------------------------------------------

class BattleCommandScorer(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_feat_dim=FEATURE_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.ActionEncoder = ActionEncoder()
        # state projection network
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, embed_dim), nn.ReLU(),
        )
        # project precomputed features
        self.action_proj = nn.Sequential(
            nn.Linear(action_feat_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # If user passed in raw dicts, encode & pad them here:
        if isinstance(actions, list):
            # 1) Use your ActionEncoder to get [K,14] feature Tensors
            action_tensors = [self.ActionEncoder([ad])[0]  # or however you call it
                              for ad in actions]
            # 2) Pad into [K,14] → then add batch dim [1,K,14]
            actions = pad_sequence(action_tensors, batch_first=True, padding_value=0.0)
            actions = actions.unsqueeze(0)
        # if someone passed a single vector, make it a batch of 1
        if states.dim() == 1:
            states = states.unsqueeze(0)   # [S] → [1, S]
        # states: [B, S], actions: [B, K, F], mask: [B, K]
        B, K, F = actions.shape
        s_emb = self.state_net(states)                        # [B, H]
        flat_a = actions.view(B * K, F)                       # [B*K, F]
        flat_h = self.action_proj(flat_a)                     # [B*K, H]
        a_emb = flat_h.view(B, K, -1)                         # [B, K, H]
        s_exp = s_emb.unsqueeze(1).expand(-1, K, -1)           # [B, K, H]
        scores = (s_exp * a_emb).sum(dim=2)                   # [B, K]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-1e9"))
        return scores


class BattleTurnDataset(torch.utils.data.Dataset):
    """
    Loads per-turn data from export/<game_id>/ with precomputed features.
    """
    def __init__(self, log_csv_path: str, data_dir: str):
        self.df = pd.read_csv(log_csv_path).dropna(subset=["reward"]).reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        game_id  = str(row.game_id)
        prefix   = row.state_prefix
        base     = os.path.join(self.data_dir, game_id)
        # load state
        bf = np.load(os.path.join(base, f"battlefield_tensor_{prefix}.npy"))
        cid= np.load(os.path.join(base, f"creature_id_tensor_{prefix}.npy"))
        fid= np.load(os.path.join(base, f"faction_id_tensor_{prefix}.npy"))
        state = np.concatenate([bf.flatten(), cid.flatten(), fid.flatten()]).astype(np.float32)
        # load actions
        actions = np.load(os.path.join(base, f"action_feats_{prefix}.npy")).astype(np.float32)
        chosen_idx = int(open(os.path.join(base, f"chosen_idx_{prefix}.txt")).read())
        reward     = float(row.reward)
        return {
            "state":      torch.from_numpy(state),
            "actions":    torch.from_numpy(actions),
            "chosen_idx": torch.tensor(chosen_idx, dtype=torch.long),
            "reward":     torch.tensor(reward, dtype=torch.float32),
        }
