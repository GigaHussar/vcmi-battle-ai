import json
import numpy as np
from pathlib import Path
import pandas as pd
import os
import sys

def encode_battle_state_from_json(battle_data):
    """
    Converts battle_data (parsed JSON) into tensors:
    - battlefield_tensor: [11, 15, 16]
    - creature_id_matrix: [11, 15]
    - faction_id_matrix: [11, 15]
    """
    HEIGHT, WIDTH_PLAYABLE, WIDTH_FULL = 11, 15, 17
    tensor_channels = 16
    tensor_battlefield = np.zeros((HEIGHT, WIDTH_PLAYABLE, tensor_channels), dtype=np.float32)
    creature_id_matrix = np.zeros((HEIGHT, WIDTH_PLAYABLE), dtype=np.int32)
    faction_id_matrix = np.zeros((HEIGHT, WIDTH_PLAYABLE), dtype=np.int32)

    CHANNEL = {
        "count": 0, "side": 1, "obstacle": 2, "can_retaliate": 3, "canShoot": 4,
        "canCast": 5, "isShooter": 6, "is_large": 7, "has_ammo_cart": 8, "ghost": 9,
        "is_on_native_terrain": 10, "morale": 11, "luck": 12, "initiative": 13,
        "hp": 14, "unit_present": 15
    }

    def normalize_morale(val): return (val + 3) / 6
    def normalize_luck(val): return (val + 3) / 6
    def hex_to_coords(hex_id): return hex_id % WIDTH_FULL, hex_id // WIDTH_FULL

    impassable_hexes = set()
    for obstacle in battle_data.get("obstacles", []):
        impassable_hexes.update(obstacle["blocking_tiles"])

    for unit in battle_data["all_units"]:
        side = -1 if unit["side"] == "attacker" else 1
        count = unit["count"]
        for hex_id in unit["occupied_hexes"]:
            x, y = hex_to_coords(hex_id)
            if 1 <= x <= 15:
                x -= 1
                creature_id_matrix[y, x] = unit["creature_id"]
                faction_id_matrix[y, x] = unit["faction"]
                tensor_battlefield[y, x, CHANNEL["count"]] = count
                tensor_battlefield[y, x, CHANNEL["side"]] = side
                tensor_battlefield[y, x, CHANNEL["obstacle"]] = 1 if hex_id in impassable_hexes else 0
                tensor_battlefield[y, x, CHANNEL["can_retaliate"]] = float(unit["can_retaliate"])
                tensor_battlefield[y, x, CHANNEL["canShoot"]] = float(unit["canShoot"])
                tensor_battlefield[y, x, CHANNEL["canCast"]] = float(unit["canCast"])
                tensor_battlefield[y, x, CHANNEL["isShooter"]] = float(unit["isShooter"])
                tensor_battlefield[y, x, CHANNEL["is_large"]] = float(unit["is_large"])
                tensor_battlefield[y, x, CHANNEL["has_ammo_cart"]] = float(unit["has_ammo_cart"])
                tensor_battlefield[y, x, CHANNEL["ghost"]] = float(unit["ghost"])
                tensor_battlefield[y, x, CHANNEL["is_on_native_terrain"]] = float(unit["is_on_native_terrain"])
                tensor_battlefield[y, x, CHANNEL["morale"]] = normalize_morale(unit["morale"])
                tensor_battlefield[y, x, CHANNEL["luck"]] = normalize_luck(unit["luck"])
                tensor_battlefield[y, x, CHANNEL["initiative"]] = unit["initiative"]
                tensor_battlefield[y, x, CHANNEL["hp"]] = unit["hp"]
                tensor_battlefield[y, x, CHANNEL["unit_present"]] = 1

    for hex_id in impassable_hexes:
        x, y = hex_to_coords(hex_id)
        if 1 <= x <= 15:
            x -= 1
            if tensor_battlefield[y, x, CHANNEL["unit_present"]] == 0:
                tensor_battlefield[y, x, CHANNEL["obstacle"]] = 1

    return tensor_battlefield, creature_id_matrix, faction_id_matrix

# Constants
WIDTH_FULL = 17
WIDTH_PLAYABLE = 15
HEIGHT = 11

# Get base path relative to this script (vcmi/)
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
export_dir = os.path.join(base_path, "export")
h3ai_dir = os.path.join(base_path, "h3ai")
os.makedirs(export_dir, exist_ok=True)

# Path to battle state JSON file
battle_json_path = os.path.join(export_dir, "battle.json")

#Check if the script was called with a command-line argument. If yes, use it as game_id. If not, just use the string 'default'
game_id = sys.argv[1] if len(sys.argv) > 1 else "default"

def flatten_for_inspection():
    tensor_flat_df = pd.DataFrame(
        tensor_battlefield.reshape(HEIGHT * WIDTH_PLAYABLE, 16),
        columns=[
            "count", "side", "obstacle", "can_retaliate", "canShoot", "canCast",
            "isShooter", "is_large", "has_ammo_cart", "ghost", "is_on_native_terrain",
            "morale", "luck", "initiative", "hp", "unit_present"
        ]
    )

    flat_output_path = os.path.join(export_dir, "battlefield_tensor_flat.csv")
    tensor_flat_df.to_csv(flat_output_path, index=False)
    return flat_output_path




# Load the battle data
with open(battle_json_path) as f:
    battle_data = json.load(f)

# Initialize tensor of shape [HEIGHT, WIDTH_PLAYABLE, CHANNELS]
# 16 channels for scalar/binary features, creature_id and faction_id handled separately as embeddings
tensor_channels = 16
tensor_battlefield = np.zeros((HEIGHT, WIDTH_PLAYABLE, tensor_channels), dtype=np.float32)

# List of features to populate per tile (not including creature_id/faction_id)
# Channels: 0=count, 1=side, 2=obstacle, 3=can_retaliate, 4=canShoot, 5=canCast,
# 6=isShooter, 7=is_large, 8=has_ammo_cart, 9=ghost, 10=is_on_native_terrain,
# 11=normalized_morale, 12=normalized_luck, 13=initiative, 14=hp, 15=unit_present
CHANNEL = {
    "count": 0,
    "side": 1,
    "obstacle": 2,
    "can_retaliate": 3,
    "canShoot": 4,
    "canCast": 5,
    "isShooter": 6,
    "is_large": 7,
    "has_ammo_cart": 8,
    "ghost": 9,
    "is_on_native_terrain": 10,
    "morale": 11,
    "luck": 12,
    "initiative": 13,
    "hp": 14,
    "unit_present": 15
}

# Normalize helper functions
def normalize_morale(val):
    return (val + 3) / 6  # Morale ranges from -3 to +3

def normalize_luck(val):
    return (val + 3) / 6  # Luck ranges from -3 to +3

# Obstacle processing
impassable_hexes = set()
for obstacle in battle_data["obstacles"]:
    impassable_hexes.update(obstacle["blocking_tiles"])

# Coordinate helper
def hex_to_coords(hex_id):
    x = hex_id % WIDTH_FULL
    y = hex_id // WIDTH_FULL
    return x, y

# Prepare embedding index matrices for creature_id and faction
creature_id_matrix = np.zeros((HEIGHT, WIDTH_PLAYABLE), dtype=np.int32)
faction_id_matrix = np.zeros((HEIGHT, WIDTH_PLAYABLE), dtype=np.int32)

# Fill tensor from units
for unit in battle_data["all_units"]:
    side = -1 if unit["side"] == "attacker" else 1
    count = unit["count"]
    for hex_id in unit["occupied_hexes"]:
        x, y = hex_to_coords(hex_id)
        if 1 <= x <= 15:
            x -= 1  # convert to 0-based index

            # Creature and faction IDs for embeddings
            creature_id_matrix[y, x] = unit["creature_id"]
            faction_id_matrix[y, x] = unit["faction"]

            # Scalar/binary feature population
            tensor_battlefield[y, x, CHANNEL["count"]] = count
            tensor_battlefield[y, x, CHANNEL["side"]] = side
            tensor_battlefield[y, x, CHANNEL["obstacle"]] = 1 if hex_id in impassable_hexes else 0
            tensor_battlefield[y, x, CHANNEL["can_retaliate"]] = float(unit["can_retaliate"])
            tensor_battlefield[y, x, CHANNEL["canShoot"]] = float(unit["canShoot"])
            tensor_battlefield[y, x, CHANNEL["canCast"]] = float(unit["canCast"])
            tensor_battlefield[y, x, CHANNEL["isShooter"]] = float(unit["isShooter"])
            tensor_battlefield[y, x, CHANNEL["is_large"]] = float(unit["is_large"])
            tensor_battlefield[y, x, CHANNEL["has_ammo_cart"]] = float(unit["has_ammo_cart"])
            tensor_battlefield[y, x, CHANNEL["ghost"]] = float(unit["ghost"])
            tensor_battlefield[y, x, CHANNEL["is_on_native_terrain"]] = float(unit["is_on_native_terrain"])
            tensor_battlefield[y, x, CHANNEL["morale"]] = normalize_morale(unit["morale"])
            tensor_battlefield[y, x, CHANNEL["luck"]] = normalize_luck(unit["luck"])
            tensor_battlefield[y, x, CHANNEL["initiative"]] = unit["initiative"]
            tensor_battlefield[y, x, CHANNEL["hp"]] = unit["hp"]
            tensor_battlefield[y, x, CHANNEL["unit_present"]] = 1

# Add remaining obstacle-only hexes
for hex_id in impassable_hexes:
    x, y = hex_to_coords(hex_id)
    if 1 <= x <= 15:
        x -= 1
        if tensor_battlefield[y, x, CHANNEL["unit_present"]] == 0:
            tensor_battlefield[y, x, CHANNEL["obstacle"]] = 1

# Save battlefield features and ID matrices for use in model
np.save(os.path.join(h3ai_dir, f"battlefield_tensor_{game_id}.npy"), tensor_battlefield)
np.save(os.path.join(h3ai_dir, f"creature_id_tensor_{game_id}.npy"), creature_id_matrix)
np.save(os.path.join(h3ai_dir, f"faction_id_tensor_{game_id}.npy"), faction_id_matrix)

flatten_for_inspection()
pd.DataFrame(creature_id_matrix).to_csv(os.path.join(export_dir, "creature_id_matrix.csv"), index=False)
pd.DataFrame(faction_id_matrix).to_csv(os.path.join(export_dir, "faction_id_matrix.csv"), index=False)
