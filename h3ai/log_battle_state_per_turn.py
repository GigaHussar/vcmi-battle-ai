import os
import json
import time
import numpy as np
from pathlib import Path
from battle_state_to_tensor import encode_battle_state_from_json
import csv
import shutil
import pandas as pd
from predictor_helpers import extract_all_possible_commands
from model import ActionEncoder   

EXPORT_DIR = Path("/Users/syntaxerror/vcmi/export")
STATE_FILE = EXPORT_DIR / "battle.json"
ACTIONS_FILE = EXPORT_DIR / "possible_actions.json" 

# Initialize action encoder (adjust embed_dim if desired)
action_encoder = ActionEncoder(type_emb_dim=8)

def get_export_subdir(game_id):
    folder = EXPORT_DIR / f"game_{game_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def log_state(game_id, turn, performance):
    state = read_json(STATE_FILE)
    if not state:
        print("❌ No battle.json found.")
        return

    features, creature_ids, faction_ids = encode_battle_state_from_json(state)

    # Read and encode possible actions
    actions_data = read_json(ACTIONS_FILE)
    if actions_data:
        # Extract structured action dicts
        action_dicts = extract_all_possible_commands(actions_data)
        # Encode actions to feature tensor [k,14]
        action_feats = action_encoder(action_dicts).numpy()
    else:
        action_dicts = []
        action_feats = np.empty((0, action_encoder.type_emb.embedding_dim + 6))  # empty array


    export_subdir = get_export_subdir(game_id)
    TENSOR_DIR = export_subdir / "tensors"
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the battle state JSON, assign a unique name
    shutil.copy(STATE_FILE, TENSOR_DIR / f"battle_json_{game_id}_{turn}.json")
    # Copy the raw actions JSON as well
    if actions_data:
        shutil.copy(ACTIONS_FILE, TENSOR_DIR / f"actions_json_{game_id}_{turn}.json")


    
    # Save tensors
    np.save(TENSOR_DIR / f"battle_state_{game_id}_{turn}.npy", features)
    np.save(TENSOR_DIR / f"creature_id_{game_id}_{turn}.npy", creature_ids)
    np.save(TENSOR_DIR / f"faction_id_{game_id}_{turn}.npy", faction_ids)
    np.save(TENSOR_DIR / f"action_feats_{game_id}_{turn}.npy", action_feats)

    # Log performance label
    csv_path = export_subdir / "value_labels.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["game_id", "turn", "performance"])
        writer.writerow([game_id, turn, performance])

def update_value_csv_path(game_id: int, final_performance: float, csv_path: Path = None):
    if csv_path is None:
        csv_path = EXPORT_DIR / f"game_{game_id}" / "value_labels.csv"

    try:
        df = pd.read_csv(csv_path)
        if "game_id" not in df.columns or "performance" not in df.columns:
            print("❌ CSV missing required columns.")
            return False

        df.loc[df["game_id"] == game_id, "performance"] = final_performance
        df.to_csv(csv_path, index=False)
        print(f"✅ Updated performance for game_id {game_id} to {final_performance}")
        return True
    except Exception as e:
        print(f"❌ Failed to update CSV: {e}")
        return False