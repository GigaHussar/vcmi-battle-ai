import json
import numpy as np
from pathlib import Path

# Constants
WIDTH_FULL = 17
WIDTH_PLAYABLE = 15
HEIGHT = 11

# Load battle.json
with open("/mnt/data/battle.json", "r") as f:
    battle_data = json.load(f)

# Initialize the battlefield matrix
# Each cell is a tuple: (creature_id, side (0=attacker, 1=defender), count, obstacle_flag)
battlefield = [[(0, 0, 0, 0) for _ in range(WIDTH_PLAYABLE)] for _ in range(HEIGHT)]

# Mark obstacle hexes
impassable_hexes = set()
for obstacle in battle_data["obstacles"]:
    impassable_hexes.update(obstacle["blocking_tiles"])

# Helper function to convert hex ID to (x, y)
def hex_to_coords(hex_id):
    x = hex_id % WIDTH_FULL
    y = hex_id // WIDTH_FULL
    return x, y

# Place units
for unit in battle_data["all_units"]:
    side = 0 if unit["side"] == "attacker" else 1
    creature_id = unit["creature_id"]
    count = unit["count"]
    
    for hex_id in unit["occupied_hexes"]:
        x, y = hex_to_coords(hex_id)
        if 1 <= x <= 15:
            matrix_x = x - 1  # Adjust to 0-based playable width
            obstacle_flag = 1 if hex_id in impassable_hexes else 0
            battlefield[y][matrix_x] = (creature_id, side, count, obstacle_flag)

# Now fill in remaining obstacle-only tiles
for hex_id in impassable_hexes:
    x, y = hex_to_coords(hex_id)
    if 1 <= x <= 15:
        matrix_x = x - 1
        if battlefield[y][matrix_x] == (0, 0, 0, 0):
            battlefield[y][matrix_x] = (0, 0, 0, 1)

