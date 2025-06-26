import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Action encoding utilities — used by file2.py to precompute features
# Map your JSON type IDs → string labels
ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
}

# ACTION_TYPE_LABELS: list of all possible action type labels (e.g., ['move', 'attack', ...])
ACTION_TYPE_LABELS = ["defend", "wait", "move", "melee"]
# ACTION_LABEL_TO_IDX: dict mapping each action label string to a unique integer index.
ACTION_LABEL_TO_IDX = {label: i for i, label in enumerate(ACTION_TYPE_LABELS)}

# Board geometry for normalizing coords
WIDTH_FULL      = 17
WIDTH_PLAYABLE  = 15
HEIGHT          = 11
STATE_DIM       = 2970
EMBED_DIM = 64   # joint embedding size
FEATURE_DIM = 14 # precomputed action feature dimension

# hex_to_coords is a helper that converts a single number (hex index from JSON file) into x, y coordinates.
def hex_to_coords(hex_id):
    x = hex_id % WIDTH_FULL
    y = hex_id // WIDTH_FULL
    return x, y


def encode_hex_field(hex_idx):
    """
    Takes a hex index (a number identifying one tile on a grid) and converts it
    into three numbers: (normalized x, normalized y, valid flag).

    - valid flag: 1.0 if this is a real tile, 0.0 if the input was invalid
    """
    # Check if the input is missing or negative; mark invalid if so
    if hex_idx is None or hex_idx < 0:
        # Returns tensor with shape (3,)
        return torch.tensor([0.0, 0.0, 0.0])
    # Convert the single index into grid coordinates
    x, y = hex_to_coords(hex_idx)
    # Clamp x and y so they stay within the board dimensions
    x = min(max(x, 0), WIDTH_PLAYABLE - 1)
    y = min(max(y, 0), HEIGHT        - 1)
    # Normalize x and y to the range [0, 1]
    # Compose and return the final tensor of shape (3,): [x_norm, y_norm, 1.0]
    return torch.tensor([x / (WIDTH_PLAYABLE - 1), y / (HEIGHT - 1), 1.0])


class ActionEncoder(nn.Module):
    """
    Converts a list of action description dictionaries into a batch of fixed-size feature vectors.

    Each action vector has:
      - A learnable embedding for the action type (size = type_emb_dim)
      - Two encoded hex fields (each a tensor of shape (3,))

    Therefore, the final vector for each action has dimension:
      type_emb_dim + 3 + 3 = type_emb_dim + 6

    With the default type_emb_dim=8, the output dimension is 14.

    Used in file2.py to precompute feature files (.npy) for model training or analysis.
    """
    def __init__(self, type_emb_dim: int = 8):
        # Initialize the nn.Module base: this tells PyTorch to track any layers or tensors you add so they become learnable parameters that can be updated during training
        super().__init__()
        # nn.Embedding sets up a table of learnable vectors: given an action type index, it returns the corresponding embedding of size type_emb_dim
        self.type_emb = nn.Embedding(len(ACTION_TYPE_LABELS), type_emb_dim)

    def forward(self, action_dicts: list[dict]) -> torch.Tensor:
        """
        Args:
            action_dicts (list of dict): each dict has:
              - 'type': str, the action type (must be in ACTION_TYPE_LABELS)
              - 'hex1', 'hex2': optional ints (tile indices); default to -1 if missing

        Returns:
            torch.Tensor of shape (batch_size, feature_dim)
            where batch_size = len(action_dicts)
                  feature_dim = type_emb_dim + 6 (default 14)
        """
        batch = []
        for a in action_dicts:
            t_idx = ACTION_LABEL_TO_IDX[a['type']]
            e_t   = self.type_emb(torch.tensor(t_idx, device=self.type_emb.weight.device)) # shape: (type_emb_dim,)
            
            f1    = encode_hex_field(a.get("hex1", -1)) # shape: (3,)
            f2    = encode_hex_field(a.get("hex2", -1)) # shape: (3,)

            # Combine type embedding with hex field encodings
            batch.append(torch.cat([e_t, f1, f2], dim=0))
        # Stack into output tensor of shape (batch_size, feature_dim)
        return torch.stack(batch, dim=0)

class StateEncoder(nn.Module):
    """
    Projects a flat state vector into the joint embedding space.
    """
    def __init__(self, state_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, embed_dim), nn.ReLU(),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_vec: Tensor of shape (batch_size, state_dim)
        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        return self.net(state_vec)

class ActionProjector(nn.Module):
    """
    Takes raw action feature vectors (from ActionEncoder) and projects them into the same embedding space as states.

    While ActionEncoder builds the 14-dim raw features, this module transforms those into embed_dim features
    that can be directly compared (dot-product) with state embeddings.
    
    Projects precomputed action features into the same embedding space as states.
    """
    def __init__(self, feature_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, action_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_feats: Tensor of shape (batch_size, num_actions, feature_dim)
        Returns:
            Tensor of shape (batch_size, num_actions, embed_dim)
        """
        B, K, F = action_feats.shape
        flat = action_feats.view(B * K, F)
        projected = self.net(flat)  # (B*K, embed_dim)
        return projected.view(B, K, -1)  # (batch_size, num_actions, embed_dim)

class CompatibilityScorer(nn.Module):
    """
    Computes dot-product scores between state embeddings and action embeddings.
    """
    def forward(self,
                s_emb: torch.Tensor,
                a_emb: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            s_emb: Tensor of shape (batch_size, embed_dim)
            a_emb: Tensor of shape (batch_size, num_actions, embed_dim)
            mask: optional BoolTensor of shape (batch_size, num_actions)
        Returns:
            scores: Tensor of shape (batch_size, num_actions)
        """
        B, K, H = a_emb.shape
        # expand state to (B, K, H)
        s_exp = s_emb.unsqueeze(1).expand(-1, K, -1)
        # dot product over embedding dim
        scores = (s_exp * a_emb).sum(dim=2)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-1e9'))
        return scores

class BattleCommandScorer(nn.Module):
    def __init__(self,
                 state_encoder: StateEncoder,
                 action_encoder: ActionEncoder,
                 action_projector: ActionProjector,
                 compat_scorer: CompatibilityScorer):
        super().__init__()
        self.state_enc    = state_encoder
        self.action_enc   = action_encoder
        self.action_proj  = action_projector
        self.compat_scorer= compat_scorer

    def forward(self, state_vec, action_dicts):
        """
        state_vec: tensor [batch_size, ...]
        action_dicts: list/dict of raw action features, length = num_actions
        Returns: tensor [batch_size, num_actions] of scores
        """
        # 1) Embed state → [batch, D]
        s_emb = self.state_enc(state_vec)

        # 2) Embed raw actions → [num_actions, F]
        raw_feats = self.action_enc(action_dicts)

        # 3) Project actions into same D-dim space, add batch dim → [batch, num_actions, D]
        a_emb   = self.action_proj(raw_feats.unsqueeze(0).to(s_emb.device))

        # 4) Dot-product scores → [batch, num_actions]
        return self.compat_scorer(s_emb, a_emb)

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
