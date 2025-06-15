import json
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
STATE_FILE = Path("/Users/syntaxerror/vcmi/export/battle.json")

# === FEATURE ENCODING ===

def encode_battle_state(state_json):
    """
    Encodes selected fields from battle.json into a flat vector.
    Focuses on active stack and overall battle summary.
    """
    features = []

    # Army strengths
    features.append(state_json.get("army_strength_attacker", 0) / 100000.0)
    features.append(state_json.get("army_strength_defender", 0) / 100000.0)

    # Basic battlefield info
    features.append(state_json.get("terrain", 0) / 100.0)
    features.append(state_json.get("battlefield_type", 0) / 100.0)

    # Unit data
    units = state_json.get("all_units", [])
    active_stack_id = None
    for unit in units:
        if unit.get("is_active"):
            active_stack_id = unit.get("id")
            # Unit-specific features
            features.append(unit.get("count", 0) / 1000.0)
            features.append(unit.get("hp", 0) / 10000.0)
            features.append(unit.get("initiative", 0) / 100.0)
            features.append(1.0 if unit.get("canAct") else 0.0)
            features.append(1.0 if unit.get("canShoot") else 0.0)
            features.append(1.0 if unit.get("canCast") else 0.0)
            features.append(1.0 if unit.get("isShooter") else 0.0)
            features.append(unit.get("morale", 0) / 10.0)
            features.append(unit.get("luck", 0) / 10.0)
            features.append(1.0 if unit.get("isClone") else 0.0)
            break

    # Fallback if no active unit found
    if active_stack_id is None:
        features += [0.0] * 12

    # Number of alive units on both sides
    atk_alive = sum(1 for u in units if u["side"] == "attacker" and not u["is_dead"])
    def_alive = sum(1 for u in units if u["side"] == "defender" and not u["is_dead"])
    features.append(atk_alive / 20.0)
    features.append(def_alive / 20.0)

    return np.array(features, dtype=np.float32)

# === TEST ENCODER ===
def test_encoder():
    try:
        with open(STATE_FILE) as f:
            state_json = json.load(f)
        features = encode_battle_state(state_json)
        print("✅ Encoded feature vector shape:", features.shape)
        print("🔢 Sample vector:", features[:10])
    except Exception as e:
        print("❌ Failed to encode:", e)

test_encoder()
