import os
import json
import time
import numpy as np
from pathlib import Path
from battle_state_to_tensor import encode_battle_state_from_json
import csv

EXPORT_DIR = Path("/Users/syntaxerror/vcmi/export")
TENSOR_DIR = Path("h3ai")
LABELS_CSV = Path("value_labels.csv")
STATE_FILE = EXPORT_DIR / "battle.json"

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

    # Save raw battle.json for this turn
    with open(f"h3ai/battle_json_{game_id}_{turn}.json", "w") as f_out:
        json.dump(state, f_out, indent=2)
    
    # Save tensors
    np.save(TENSOR_DIR / f"battle_state_{game_id}_{turn}.npy", features)
    np.save(TENSOR_DIR / f"creature_id_{game_id}_{turn}.npy", creature_ids)
    np.save(TENSOR_DIR / f"faction_id_{game_id}_{turn}.npy", faction_ids)

    # Log performance label
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LABELS_CSV.exists()

    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["game_id", "turn", "performance"])
        writer.writerow([game_id, turn, performance])
