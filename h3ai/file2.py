import json
import numpy as np
from pathlib import Path
import pandas as pd
import logging
import torch
from model import BattleCommandScorer, ActionEncoder
import csv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root (vcmi/) two levels up from this file
base_path = Path(__file__).resolve().parent.parent

# Constants
WIDTH_FULL = 17
WIDTH_PLAYABLE = 15
HEIGHT = 11

CHANNEL = {
    "count": 0, "side": 1, "obstacle": 2, "can_retaliate": 3, "canShoot": 4,
    "canCast": 5, "isShooter": 6, "is_large": 7, "has_ammo_cart": 8, "ghost": 9,
    "is_on_native_terrain": 10, "morale": 11, "luck": 12, "initiative": 13,
    "hp": 14, "unit_present": 15
}

ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
    # 6 is now ignored
}

# ---- CONFIG ----
STATE_FILE    = base_path / "export" / "battle.json"
ACTIONS_FILE  = base_path / "export" / "possible_actions.json"
MODEL_WEIGHTS = base_path / "h3ai" / "model_weights.pth"
MASTER_LOG    = base_path / "export" / "master_log.csv"

# 1) Instantiate your model once, load weights if available
model = BattleCommandScorer()
if MODEL_WEIGHTS.exists():
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.eval()

def extract_all_possible_commands(actions_data):
    """
    Parses possible_actions.json and returns a list of structured commands:
      [
        {"type":"defend"},
        {"type":"wait"},
        {"type":"move",  "hex1": 45},
        {"type":"melee", "hex1": from_hex, "hex2": target_hex},
        ...
      ]
    """
    commands = []

    for action in actions_data.get("actions", []):
        type_id = action.get("type")
        action_type = ACTION_TYPE_MAP.get(type_id)
        if action_type is None:
            # skip unhandled types (including type 6)
            continue

        if action_type in ("defend", "wait"):
            # For "defend" and "wait", we just append the type
            commands.append({"type": action_type})

        elif action_type == "move":
            # For "move", we need to extract reachable tiles
            for tile in action.get("reachable_tiles", []):
                commands.append({
                    "type": "move",
                    "hex1": tile["hex"]
                })

        elif action_type == "melee":
            # For "melee", we need to extract hexes to move to and targets
            for target in action.get("melee_targets", []):
                if target.get("can_melee_attack", False):
                    commands.append({
                        "type": "melee",
                        "hex1": target["attack_from"]["hex"],
                        "hex2": target["hex"]
                    })

    return commands

def predict_best_command():
    try:
        # 2) Load & encode state
        state_json = json.loads(STATE_FILE.read_text())
        feats, c_ids, f_ids = encode_battle_state_from_json(state_json)
        # flatten & concat into one vector of length 2970
        state_vec = torch.from_numpy(
            np.concatenate([feats.flatten(),
                            c_ids.flatten(),
                            f_ids.flatten()])
        ).float()

        # 3) Load & extract actions
        actions_data  = json.loads(ACTIONS_FILE.read_text())
        action_dicts  = extract_all_possible_commands(actions_data)
        if not action_dicts:
            print("⚠️ No legal actions.")
            return None

        # 4) Score actions
        with torch.no_grad():
            scores = model(state_vec, action_dicts)  # [k]

        # 5) Pick best (or sample)
        best_idx = scores.argmax().item()
        chosen  = action_dicts[best_idx]

        print(f"🧠 Model chose: {chosen}")
        return chosen

    except Exception as e:
        print("❌ Predictor failed:", e)
        return None
    
# Normalize morale and luck
def normalize_stat(val: float, low: float = -3, high: float = +3) -> float:
    """Scale a stat in [low, high] to [0, 1]."""
    return (val - low) / (high - low)

def hex_to_coords(hex_id: int) -> tuple[int, int]:
    """
    Convert a raw “full-grid” index into playable (x, y):
      - Full grid is WIDTH_FULL columns (0 to 16).
      - Only columns 1–15 are actually playable.
      - We subtract 1 so that playable columns map to x=0..14.
    """
    x_full = hex_id % WIDTH_FULL
    y      = hex_id // WIDTH_FULL
    return x_full - 1, y

def encode_battle_state_from_json(battle_data: dict):
    """
    Returns (tensor, creature_ids, faction_ids):
      - tensor:   [HEIGHT, WIDTH_PLAYABLE, 16]
      - creature_ids: [HEIGHT, WIDTH_PLAYABLE]
      - faction_ids:  [HEIGHT, WIDTH_PLAYABLE]
    """
    tensor = np.zeros((HEIGHT, WIDTH_PLAYABLE, len(CHANNEL)), dtype=np.float32)
    creature_ids = np.zeros((HEIGHT, WIDTH_PLAYABLE), dtype=np.int32)
    faction_ids  = np.zeros((HEIGHT, WIDTH_PLAYABLE), dtype=np.int32)

    # gather all obstacle-hexes
    blocking = {
        h
        for obst in battle_data.get("obstacles", [])
        for h in obst.get("blocking_tiles", [])
    }

    for unit in battle_data.get("all_units", []):
        side_val = -1 if unit["side"] == "attacker" else +1
        for hex_id in unit["occupied_hexes"]:
            x, y = hex_to_coords(hex_id)
            if not (0 <= x < WIDTH_PLAYABLE): 
                continue

            creature_ids[y, x] = unit["creature_id"]
            faction_ids[y, x]  = unit["faction"]

            cell = tensor[y, x]
            cell[CHANNEL["count"]] = unit["count"]
            cell[CHANNEL["side"]]  = side_val
            cell[CHANNEL["obstacle"]] = float(hex_id in blocking)

            # batch‐fill all boolean flags
            for flag in (
                "can_retaliate","canShoot","canCast","isShooter",
                "is_large","has_ammo_cart","ghost","is_on_native_terrain"
            ):
                cell[CHANNEL[flag]] = float(unit.get(flag, False))

            cell[CHANNEL["morale"]]      = normalize_stat(unit["morale"])
            cell[CHANNEL["luck"]]        = normalize_stat(unit["luck"])
            cell[CHANNEL["initiative"]] = unit["initiative"]
            cell[CHANNEL["hp"]]         = unit["hp"]
            cell[CHANNEL["unit_present"]] = 1.0

    # mark obstacle-only tiles
    for hex_id in blocking:
        x, y = hex_to_coords(hex_id)
        if 0 <= x < WIDTH_PLAYABLE and tensor[y, x, CHANNEL["unit_present"]] == 0:
            tensor[y, x, CHANNEL["obstacle"]] = 1.0

    return tensor, creature_ids, faction_ids

def flatten_for_inspection(tensor: np.ndarray, csv_path: Path):
    """Save [H*W, C] flattened view of `tensor` to CSV (column names from CHANNEL)."""
    df = pd.DataFrame(
        tensor.reshape(-1, tensor.shape[-1]),
        columns=list(CHANNEL.keys())
    )
    df.to_csv(csv_path, index=False)
    logger.info(f"Flattened tensor saved to {csv_path}")

def save_battle_state_to_tensors(game_id: str, base_path: Path):
    """
    Loads battle.json, encodes the state once, and then
    saves .npy tensors plus CSV inspection files.
    """
    export_dir        = base_path / "export"
    h3ai_dir          = base_path / "h3ai"
    battle_json_path  = export_dir / "battle.json"
    export_dir.mkdir(exist_ok=True)
    h3ai_dir.mkdir(exist_ok=True)

    if not battle_json_path.exists():
        logger.error(f"No battle.json found at {battle_json_path}")
        return   
    battle_data = json.loads(battle_json_path.read_text())

    tensor, creature_ids, faction_ids = encode_battle_state_from_json(battle_data)

    # npy outputs
    np.save(h3ai_dir / f"battlefield_tensor_{game_id}.npy", tensor)
    np.save(h3ai_dir / f"creature_id_tensor_{game_id}.npy", creature_ids)
    np.save(h3ai_dir / f"faction_id_tensor_{game_id}.npy", faction_ids)
    logger.info(f"Saved .npy tensors with game_id={game_id}")

    # CSV inspection
    flatten_for_inspection(tensor, export_dir / f"battlefield_tensor_flat_{game_id}.csv")
    pd.DataFrame(creature_ids).to_csv(export_dir / f"creature_id_matrix_{game_id}.csv", index=False)
    pd.DataFrame(faction_ids).to_csv(export_dir / f"faction_id_matrix_{game_id}.csv", index=False)
    logger.info("Saved CSV inspection files")

def save_action_tensor(game_id: str, turn: int, action_dicts: list[dict], out_dir: Path):
    # action_dicts is the list you got from extract_all_possible_commands(...)
    enc = ActionEncoder()
    # enc returns a torch.Tensor of shape [k, feature_dim] (e.g. [k,14])
    action_tensor = enc(action_dicts).cpu().numpy()
    out_path = out_dir / f"action_feats_{game_id}_{turn}.npy"
    np.save(out_path, action_tensor)
    return out_path

def save_chosen_index(game_id: str, turn: int, chosen_idx: int, out_dir: Path):
    idx_path = out_dir / f"chosen_idx_{game_id}_{turn}.txt"
    idx_path.write_text(str(chosen_idx))
    return idx_path

def init_master_log():
    """
    Create the CSV file with headers if it doesn’t already exist.
    """
    if not MASTER_LOG.exists():
        MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(MASTER_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "game_id",
                "turn",
                "state_prefix",
                "action_prefix",
                "reward"
            ])

def log_turn_to_csv(game_id: str, turn: int):
    """
    Append one row with the given game_id and turn.
    state_prefix and action_prefix are both <game_id>_<turn>.
    Reward is left blank for now.
    """
    init_master_log()
    prefix = f"{game_id}_{turn}"
    with open(MASTER_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            game_id,
            turn,
            prefix,
            prefix,
            ""        # reward placeholder
        ])

def fill_battle_rewards(game_id: str, final_score: float):
    """
    After battle ends, back-fill the 'reward' column for all rows
    matching this game_id.
    """
    df = pd.read_csv(MASTER_LOG)
    mask = df["game_id"] == game_id
    df.loc[mask, "reward"] = final_score
    df.to_csv(MASTER_LOG, index=False)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Encode & export a Heroes3 battle state to tensors and CSV."
    )
    parser.add_argument(
        "--game-id", "-g",
        default="default",
        help="Identifier for this battle run (used in output filenames)"
    )
    args = parser.parse_args()
    save_battle_state_to_tensors(args.game_id, base_path)

if __name__ == "__main__":
    main()
