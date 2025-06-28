"""Utility helpers for saving encoded actions & chosen index."""
from pathlib import Path
import numpy as np

from model import ActionEncoder

_enc = ActionEncoder()

def save_action_tensor(game_id: str, turn: int, action_dicts: list[dict], out_dir: Path):
    arr = _enc(action_dicts).cpu().numpy()         # (K, 14)
    path = out_dir / f"action_feats_{game_id}_{turn}.npy"
    np.save(path, arr)
    return path

def save_chosen_index(game_id: str, turn: int, idx: int, out_dir: Path):
    path = out_dir / f"chosen_idx_{game_id}_{turn}.txt"
    path.write_text(str(idx))
    return path
