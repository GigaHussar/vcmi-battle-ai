import json
import socket

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


def get_army_strengths(state):
    return (
        state.get("army_strength_attacker", 0),
        state.get("army_strength_defender", 0)
    )

def compute_performance(kills: float, losses: float, init_def: float, init_att: float) -> float:
    """
    Returns the fraction of total casualties that were enemy kills.
    If there were no casualties, returns 0.0.
    """
    casualties = kills + losses
    underdog = init_def / (init_att + init_def)
    if casualties <= 0:
        return 0.0
    return (kills / casualties) * underdog

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

    # root-level legality (defaults to True for old logs)
    global_can_wait = actions_data.get("can_wait", True)

    for action in actions_data.get("actions", []):
        type_id = action.get("type")
        action_type = ACTION_TYPE_MAP.get(type_id)
        if action_type is None:
            # skip unhandled types (including type 6)
            continue

        if action_type == "wait":
            # prefer per-action flag if present, else use the root flag
            can_wait = action.get("can_wait", global_can_wait)
            if can_wait:
                commands.append({"type": "wait"})
                print("waiting available")
            else:
                print("waiting not available")
            continue

        elif action_type == "defend":
            commands.append({"type": "defend"})

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
