from pathlib import Path
import numpy as np
import torch
from model import ActionEncoder

_enc = ActionEncoder()
_enc.eval()                       # gradients off for this helper

@torch.no_grad()                  # same idea, but at function level
def save_action_tensor(game_id: str, turn: int,
                       action_dicts: list[dict], out_dir: Path):
    arr = _enc(action_dicts).detach().cpu().numpy()   #  ← detach !
    path = out_dir / f"action_feats_{game_id}_{turn}.npy"
    np.save(path, arr)
    return path

def save_chosen_index(game_id: str, turn: int, idx: int, out_dir: Path):
    path = out_dir / f"chosen_idx_{game_id}_{turn}.txt"
    path.write_text(str(idx))
    return path
