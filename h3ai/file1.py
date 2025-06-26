import json
import socket
import time
from pathlib import Path
from _helpers_do_not_touch import encode_battle_state_from_json, fill_battle_rewards, log_turn_to_csv, save_battle_state_to_tensors,  extract_all_possible_commands
from file2 import save_action_tensor, save_chosen_index
from model import ActionEncoder, CompatibilityScorer, ActionProjector, StateEncoder, BattleCommandScorer, STATE_DIM, EMBED_DIM, FEATURE_DIM
import torch
import numpy as np
from h3ai._runvcmi_do_not_touch import open_vcmi_process, close_vcmi_process
from h3ai._paths_do_not_touch import MODEL_WEIGHTS, EXPORT_DIR, BATTLE_JSON_PATH, ACTIONS_FILE, MASTER_LOG
import re
import shutil
import pandas as pd
from _helpers_do_not_touch import format_command_for_vcmi, read_json, send_command, get_army_strengths, organize_export_files, compute_performance

CHECK_INTERVAL = 2

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
    game_id = int(time.time())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_encoder = StateEncoder(STATE_DIM, EMBED_DIM).to(device)
    action_encoder = ActionEncoder().to(device)
    action_projector = ActionProjector(FEATURE_DIM, EMBED_DIM).to(device)
    scorer = CompatibilityScorer().to(device)
    # wrap, load weights, switch to eval
    model = BattleCommandScorer(state_encoder, action_encoder, action_projector, scorer)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device).eval()

    print("starting battle loop...")
    while True:
        state = read_json(BATTLE_JSON_PATH)     # dict with current battle state, or None if missing
        actions_data = read_json(ACTIONS_FILE)  # dict listing possible commands, turn number, etc.

        if not state or not actions_data:
            print("⏳ Waiting for battle state...")
            time.sleep(CHECK_INTERVAL)
            continue

        #    Convert JSON state into feature arrays
        #    encode_battle_state_from_json returns:
        #      feats: 2D array of state features
        #      c_ids: IDs for creatures
        #      f_ids: IDs for features
        feats, c_ids, f_ids = encode_battle_state_from_json(state)
        #    Build a single flat state vector for the model
        #    - Flatten each array and concatenate into one 1D numpy array
        #    - Convert to a PyTorch float tensor
        state_vec = torch.from_numpy(
            np.concatenate([feats.flatten(), c_ids.flatten(), f_ids.flatten()])
        ).float().unsqueeze(0).to(device) # shape: (state_dim,)

        s_emb = state_encoder(state_vec)  # (1, EMBED_DIM)

        current_turn = actions_data.get("turn", -1)

        #    Extract candidate actions as list of dicts
        #    Each dict has keys like 'type', 'hex1', 'hex2'
        action_dicts = extract_all_possible_commands(actions_data)

        # Turn the list of raw action dicts into a [1, num_actions, FEATURE_DIM] tensor
        action_feats = action_encoder(action_dicts)          # → [num_actions, FEATURE_DIM]
        action_feats = action_feats.unsqueeze(0).to(device)  # → [1, num_actions, FEATURE_DIM]
        
        a_emb = action_projector(action_feats)            # (1, K, EMBED_DIM)

        # Score actions using the pre-loaded model
        scores = scorer(s_emb, a_emb)  # shape: (1, num_actions)

        # Choose the best action (highest score)
        if action_dicts:
            # .argmax finds index of max score; .item() turns tensor->Python int
            chosen_idx = int(scores.argmax().item())
        else:
            chosen_idx = None

        # 3. save them *after* you’ve computed both names
        save_chosen_index(game_id, current_turn, chosen_idx, EXPORT_DIR)
        save_battle_state_to_tensors(f"{game_id}_{current_turn}", EXPORT_DIR)
        save_action_tensor(game_id, current_turn, action_dicts, EXPORT_DIR)
        log_turn_to_csv(game_id, current_turn)

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

        # compute raw kills & losses and performance
        kills   = initial_defender_strength - final_defender_strength
        losses  = initial_attacker_strength - final_attacker_strength
        performance = compute_performance(kills, losses)

        print("📊 Final Result:")
        print(f"   🛡  Attacker: {initial_attacker_strength} → {final_attacker_strength} (losses={losses})")
        print(f"   ⚔️  Defender: {initial_defender_strength} → {final_defender_strength} (kills={kills})")
        print(f"   🎯 Performance (kills/(kills+losses)): {performance:.3f}")
        fill_battle_rewards(game_id, performance)
        print("✅ Battle data saved.")
    else:
        print("⚠️ Could not read final state.")

    organize_export_files()
    close_vcmi_process()
    time.sleep(1)

# Run a single battle loop; change 'num_battles' to run multiple battles
num_battles = 1
for i in range(num_battles):
    battle_loop()