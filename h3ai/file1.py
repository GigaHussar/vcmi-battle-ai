import os
import json
import socket
import time
import csv
from pathlib import Path
from file2 import encode_battle_state_from_json, fill_battle_rewards, log_turn_to_csv, predict_best_command, save_battle_state_to_tensors, save_action_tensor, save_chosen_index, extract_all_possible_commands
from torch.distributions import Categorical
from model import BattleCommandScorer
import torch
import numpy as np
from runvcmi import open_vcmi_process, close_vcmi_process
from paths import EXPORT_DIR, BATTLE_JSON_PATH, ACTIONS_FILE, MASTER_LOG
import re
import shutil

# === CONFIGURATION ===
SOCKET_PORT = 5000
SOCKET_HOST = "localhost"
CHECK_INTERVAL = 2

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

def compute_performance(kills: float, losses: float) -> float:
    """
    Returns the fraction of total casualties that were enemy kills.
    If there were no casualties, returns 0.0.
    """
    total = kills + losses
    if total <= 0:
        return 0.0
    return kills / total

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

def battle_loop():
    open_vcmi_process()
    time.sleep(3)
    send_command("open_load_menu")
    time.sleep(1)
    send_command("lobby_accept")
    time.sleep(1)
    send_command("move_active_hero_right")
    time.sleep(1)

    print("🧠 Agent started. Waiting for battle to begin...")
    current_turn = 0
    last_turn = -1
    initial_attacker_strength = None
    initial_defender_strength = None
    prev_att_str = prev_def_str = None
    game_id = int(time.time())

    print("starting battle loop...")
    while True:
        state = read_json(BATTLE_JSON_PATH)
        actions_data = read_json(ACTIONS_FILE)

        if not state or not actions_data:
            print("⏳ Waiting for battle state...")
            time.sleep(CHECK_INTERVAL)
            continue
        
        feats, c_ids, f_ids = encode_battle_state_from_json(state)
        state_vec = torch.from_numpy(
            np.concatenate([feats.flatten(), c_ids.flatten(), f_ids.flatten()])
        ).float()

        current_turn = actions_data.get("turn", -1)

        # 1. get candidate actions
        action_dicts = extract_all_possible_commands(actions_data)

        # 2. encode and score them
        # 2. encode and score them
        scores = BattleCommandScorer()(state_vec, action_dicts)
        if action_dicts:
            chosen_idx = int(scores.argmax().item())
        else:
            chosen_idx = None
        # 3. save them *after* you’ve computed both names
        save_battle_state_to_tensors(f"{game_id}_{current_turn}", EXPORT_DIR)
        save_action_tensor(game_id, current_turn, action_dicts, EXPORT_DIR)
        if action_dicts:
            send_command(format_command_for_vcmi(action_dicts[chosen_idx]))
        else:
            print("⚠️ No valid command predicted; skipping this turn.")


        if last_turn == current_turn:
            print("🏁 Battle ended. Collecting result...")
            break

        last_turn = current_turn

        if initial_attacker_strength is None:
            initial_attacker_strength, initial_defender_strength = get_army_strengths(state)

        time.sleep(CHECK_INTERVAL)

    # After battle ends
    state = read_json(BATTLE_JSON_PATH)
    if state:
        final_attacker_strength, final_defender_strength = get_army_strengths(state)

        # compute raw kills & losses
        kills   = initial_defender_strength - final_defender_strength
        losses  = initial_attacker_strength - final_attacker_strength

        # new “performance” metric
        performance = compute_performance(kills, losses)

        print("📊 Final Result:")
        print(f"   🛡  Attacker: {initial_attacker_strength} → {final_attacker_strength} (losses={losses})")
        print(f"   ⚔️  Defender: {initial_defender_strength} → {final_defender_strength} (kills={kills})")
        print(f"   🎯 Performance (kills/(kills+losses)): {performance:.3f}")
        fill_battle_rewards(game_id, performance)
        print("✅ Battle data saved.")
    else:
        print("⚠️ Could not read final state.")

    # Increment battle counter after each battle
    global battle_counter
    battle_counter += 1  
    organize_export_files()
    close_vcmi_process()
    time.sleep(1)
# Run a single battle loop; change 'num_battles' to run multiple battles
num_battles = 1
for i in range(num_battles):
    battle_loop()