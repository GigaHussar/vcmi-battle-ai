import json
import numpy as np
from pathlib import Path
import pandas as pd

import torch
from model import ActionEncoder
import csv
from h3ai._paths_do_not_touch import MASTER_LOG, BASE_PATH, EXPORT_DIR, H3AI_DIR, BATTLE_JSON_PATH




def save_action_tensor(game_id: str, turn: int, action_dicts: list[dict], out_dir: Path):
    # action_dicts is the list you got from extract_all_possible_commands(...)
    enc = ActionEncoder()
    # enc returns a torch.Tensor of shape [k, feature_dim] (e.g. [k,14])
    action_tensor = enc(action_dicts).detach().cpu().numpy()
    out_path = out_dir / f"action_feats_{game_id}_{turn}.npy"
    np.save(out_path, action_tensor)
    return out_path

def save_chosen_index(game_id: str, turn: int, chosen_idx: int, out_dir: Path):
    idx_path = out_dir / f"chosen_idx_{game_id}_{turn}.txt"
    idx_path.write_text(str(chosen_idx))
    return idx_path







