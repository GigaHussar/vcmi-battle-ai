import json
import socket
import shutil
import re
from _paths_do_not_touch import EXPORT_DIR, BATTLE_JSON_PATH, H3AI_DIR, MASTER_LOG
import numpy as np
from pathlib import Path
import pandas as pd
import csv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_command_for_vcmi(action: dict) -> str:
    """
    Turn one of your action dicts into the VCMI CLI string.
    """
    t = action["type"]
    if t == "wait" or t == "defend":
        # VCMI accepts the same keyword
        return t
    elif t == "move":
        # e.g. "move 42"
        return f"move {action['hex1']}"
    elif t == "melee":
        # note: original JSON -> dict stored {"hex1":attack_from, "hex2":target}
        # but VCMI wants "melee <target_hex> <from_hex>"
        return f"melee {action['hex2']} {action['hex1']}"
    else:
        raise ValueError(f"Unknown action type `{t}`")
    
def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

SOCKET_PORT = 5000
SOCKET_HOST = "localhost"

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SOCKET_HOST, SOCKET_PORT))
            s.sendall(command.encode("utf-8"))
    except Exception as e:
        print(f"Socket error: {e}")

def organize_export_files():
    """
    Scan EXPORT_DIR for files matching *_<gameid>_*, 
    create subfolders under EXPORT_DIR if needed,
    and move them there.
    """
    if EXPORT_DIR is None:
        raise RuntimeError("EXPORT_DIR is not set; call create_export_directory() first")

    pattern = re.compile(r'_(\d+)_')
    for file in EXPORT_DIR.iterdir():
        if not file.is_file():
            continue

        match = pattern.search(file.name)
        if not match:
            continue

        game_id = match.group(1)
        dest_dir = EXPORT_DIR / game_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(str(file), str(dest_dir / file.name))

def get_army_strengths(state):
    return (
        state.get("army_strength_attacker", 0),
        state.get("army_strength_defender", 0)
    )

def compute_performance(kills: float, losses: float) -> float:
    """
    Returns the fraction of total casualties that were enemy kills.
    If there were no casualties, returns 0.0.
    """
    total = kills + losses
    if total <= 0:
        return 0.0
    return kills / total

ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
    # 6 is now ignored
}

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

def save_battle_state_to_tensors(game_id: str, base_path: Path):
    """
    Loads battle.json, encodes the state once, and then
    saves .npy tensors plus CSV inspection files.
    """
    EXPORT_DIR.mkdir(exist_ok=True)
    H3AI_DIR.mkdir(exist_ok=True)

    if not BATTLE_JSON_PATH.exists():
        logger.error(f"No battle.json found at {BATTLE_JSON_PATH}. Please run a battle first.")
        return   
    battle_data = json.loads(BATTLE_JSON_PATH.read_text())

    tensor, creature_ids, faction_ids = encode_battle_state_from_json(battle_data)

    # npy outputs
    np.save(EXPORT_DIR / f"battlefield_tensor_{game_id}.npy", tensor)
    np.save(EXPORT_DIR / f"creature_id_tensor_{game_id}.npy", creature_ids)
    np.save(EXPORT_DIR / f"faction_id_tensor_{game_id}.npy", faction_ids)
    logger.info(f"Saved .npy tensors with game_id={game_id}")

def fill_battle_rewards(game_id: str, final_score: float):
    """
    After battle ends, back-fill the 'reward' column for all rows
    matching this game_id.
    """
    df = pd.read_csv(MASTER_LOG)
    mask = df["game_id"] == game_id
    df.loc[mask, "reward"] = final_score
    df.to_csv(MASTER_LOG, index=False)

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