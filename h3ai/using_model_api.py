import json
import subprocess
from typing import Optional
from _paths_do_not_touch import EXPORT_DIR, ACTIONS_FILE, BATTLE_JSON_PATH  # Assumes this is defined in your environment
import requests

# Configurable model
MODEL_CONFIG = {
    "backend": "ollama",
    "model": "gemma3:4b",
    "api_url": "http://localhost:11434/api/generate"
}

def extract_available_actions() -> list[str]:
    """
    Generate all available VCMI CLI-compatible action strings from a VCMI battle JSON export.
    """

    if not ACTIONS_FILE.exists():
        print("Missing possible_actions.json")
        return []

    with open(ACTIONS_FILE, "r") as f:
        data = json.load(f)

    
    actions = []
    origin_hex = data.get("origin", {}).get("hex")

    for entry in data.get("actions", []):
        t = entry.get("type")

        if t == 0:  # defend
            actions.append("defend")
        elif t == 1:  # wait
            actions.append("wait")
        elif t == 4:  # move
            for tile in entry.get("reachable_tiles", []):
                actions.append(f"move {tile['hex']}")
        elif t in (5, 6):  # melee
            for target in entry.get("melee_targets", []):
                if target.get("can_melee_attack"):
                    actions.append(f"melee {target['hex']} {origin_hex}")

    return actions

def summarize_battle_state() -> str:
    """
    Summarize a VCMI battle state JSON into a compact, model-friendly string.
    If battle_json is None, loads from BATTLE_JSON_PATH.
    """
    if not BATTLE_JSON_PATH.exists():
        print("Missing battle.json")
        return ""

    with open(BATTLE_JSON_PATH, "r") as f:
        battle_json = json.load(f)
    
    def format_stack(unit):
        status = []
        if unit["is_active"]:
            status.append("active")
        if unit["is_dead"]:
            status.append("dead")
        if unit["isClone"]:
            status.append("clone")
        if unit["canShoot"]:
            status.append("shooter")
        if unit["canCast"]:
            status.append("caster")

        side = "atk" if unit["side"] == "attacker" else "def"
        base = f"{side}[{unit['id']}]: {unit['count']}x {unit['unit_name']} (HP:{unit['hp']})"
        if status:
            base += " [" + ",".join(status) + "]"
        return base

    turn = battle_json.get("_turn", 0)
    current_side = battle_json.get("current_turn_side", "unknown")
    atk_power = battle_json.get("army_strength_attacker", "?")
    def_power = battle_json.get("army_strength_defender", "?")
    location = battle_json.get("location", ["?", "?", "?"])
    battlefield = battle_json.get("battlefield_info", {}).get("identifier", "unknown")

    units = battle_json.get("all_units", [])
    attackers = [u for u in units if u["side"] == "attacker"]
    defenders = [u for u in units if u["side"] == "defender"]

    summary = [
        f"Turn {turn}, {current_side}'s move.",
        f"Map: {battlefield} at {location}, Terrain: {battle_json.get('terrain')}",
        f"Army Strengths - atk: {atk_power}, def: {def_power}",
        f"Attacker stacks: " + " | ".join(format_stack(u) for u in attackers),
        f"Defender stacks: " + " | ".join(format_stack(u) for u in defenders),
    ]

    return "\n".join(summary)

def query_gemma3_with_battle_state() -> Optional[str]:
    """
    Gathers context and actions, queries local Gemma3:4b model via Ollama HTTP API, logs reasoning in VCMI's export dir,
    and returns the chosen action for sending to the game.
    """
    from_path_summary = summarize_battle_state()
    available_actions = extract_available_actions()

    if not available_actions:
        print("No available actions.")
        return None

    prompt = (
        f"You are controlling a stack in a Heroes III style battle. Choose the best action.\n\n"
        f"Battle Summary:\n{from_path_summary}\n\n"
        f"Available Actions:\n{available_actions}\n\n"
        f"Respond in the format:\n"
        f"CHOSEN_ACTION: <one action from the list>\nREASON: <brief explanation>"
    )

    try:
        response = requests.post(
            MODEL_CONFIG["api_url"],
            json={
                "model": MODEL_CONFIG["model"],
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        output_text = result.get("response", "").strip()
    except requests.exceptions.Timeout:
        print("Ollama API call timed out.")
        return None
    except Exception as e:
        print(f"Error querying Ollama API: {e}")
        return None

    chosen_action = None
    reason = ""
    for line in output_text.splitlines():
        if line.startswith("CHOSEN_ACTION:"):
            chosen_action = line.replace("CHOSEN_ACTION:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    if not chosen_action:
        print("Model response did not include a chosen action.")
        return None

    # Log path
    log_path = EXPORT_DIR / "gemma_turn_logs.json"
    log_entry = {
        "chosen_action": chosen_action,
        "reason": reason,
        "available_actions": available_actions,
        "summary": from_path_summary
    }

    try:
        logs = {}
        if log_path.exists():
            with open(log_path, "r") as f:
                logs = json.load(f)
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Failed to log reasoning: {e}")

    return chosen_action