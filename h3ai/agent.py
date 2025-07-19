"""
Battle‑playing agent:
   • encodes current battle state + every legal action
   • feeds them through trained network
   • selects action with highest predicted performance
   • sends it back to VCMI
"""
import time
import torch
import numpy as np
from using_model_api import summarize_battle_state, extract_available_actions, query_gemma3_with_battle_state
from model import StateActionValueNet, ActionEncoder
from _helpers_do_not_touch import (
    encode_battle_state_from_json, extract_all_possible_commands,
    format_command_for_vcmi, read_json, send_command,
    get_army_strengths, compute_performance, organize_export_files,
    save_battle_state_to_tensors, log_turn_to_csv, fill_battle_rewards
)
from file2 import save_action_tensor, save_chosen_index
from _paths_do_not_touch import (
    MODEL_WEIGHTS, EXPORT_DIR, BATTLE_JSON_PATH, ACTIONS_FILE
)
from _runvcmi_do_not_touch import open_vcmi_process, close_vcmi_process
from online_finetune import fine_tune_after_battle

CHECK_INTERVAL = 2


def battle_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = StateActionValueNet().to(device)
    net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    net.eval()

    act_enc = ActionEncoder().to(device)

    open_vcmi_process()
    time.sleep(5)
    send_command("open_load_menu")
    time.sleep(1)
    send_command("lobby_accept")
    time.sleep(2)
    send_command("move_active_hero_right")
    time.sleep(2)

    game_id = int(time.time())
    turn = 0
    last_turn = -1
    init_att = init_def = None
    print("🧠 agent online")

    while True:
        state_json = read_json(BATTLE_JSON_PATH)
        actions_json = read_json(ACTIONS_FILE)
        if not state_json or not actions_json:
            print("…waiting"); time.sleep(CHECK_INTERVAL); continue

        chosen_action = query_gemma3_with_battle_state()

        if not chosen_action:
            print("⚠️ No action returned by Gemma"); break

        # Send chosen action to VCMI
        print(f"Sending chosen action to VCMI: {chosen_action}")
        send_command(chosen_action)

        # end‑of‑battle detection -------------------------------------------
        if last_turn == turn:
            time.sleep(30)
            print("⚠️ No turn change detected, assuming battle ended")
            break
        last_turn = turn
        if init_att is None:
            init_att, init_def = get_army_strengths(state_json)
        time.sleep(CHECK_INTERVAL)


    
    close_vcmi_process()

    time.sleep(2)
    
    organize_export_files()

for i in range (50000):
    battle_loop()
    print(i)
