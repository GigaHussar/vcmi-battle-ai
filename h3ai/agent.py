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

from model import StateActionValueNet, ActionEncoder
from _helpers_do_not_touch import (
    encode_battle_state_from_json, extract_all_possible_commands,
    format_command_for_vcmi, read_json, send_command,
    get_army_strengths, compute_performance, organize_export_files,
    save_battle_state_to_tensors, log_turn_to_csv
)
from file2 import save_action_tensor, save_chosen_index
from _paths_do_not_touch import (
    MODEL_WEIGHTS, EXPORT_DIR, BATTLE_JSON_PATH, ACTIONS_FILE
)
from _runvcmi_do_not_touch import open_vcmi_process, close_vcmi_process

CHECK_INTERVAL = 2


def battle_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = StateActionValueNet().to(device)
    net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    net.eval()

    act_enc = ActionEncoder().to(device)

    open_vcmi_process()
    time.sleep(5)
    send_command("open_load_menu"); time.sleep(1)
    send_command("lobby_accept"); time.sleep(1)

    game_id = int(time.time())
    last_turn = -1
    init_att = init_def = None
    print("🧠 agent online")

    while True:
        state_json = read_json(BATTLE_JSON_PATH)
        actions_json = read_json(ACTIONS_FILE)
        if not state_json or not actions_json:
            print("…waiting"); time.sleep(CHECK_INTERVAL); continue

        feats, c_ids, f_ids = encode_battle_state_from_json(state_json)
        state_vec = torch.from_numpy(
            np.concatenate([feats.flatten(), c_ids.flatten(), f_ids.flatten()])
        ).float().unsqueeze(0).to(device)

        turn = actions_json.get("turn", -1)
        actions = extract_all_possible_commands(actions_json)
        if not actions:
            print("⚠️  no valid moves"); break

        action_feats = act_enc(actions).unsqueeze(0).to(device)   # (1, K, F)
        scores = net(state_vec, action_feats)
        chosen_idx = int(scores.argmax().item())

        # logging & IO -------------------------------------------------------
        save_chosen_index(game_id, turn, chosen_idx, EXPORT_DIR)
        save_battle_state_to_tensors(f"{game_id}_{turn}", EXPORT_DIR)
        save_action_tensor(game_id, turn, actions, EXPORT_DIR)
        log_turn_to_csv(game_id, turn)
        send_command(format_command_for_vcmi(actions[chosen_idx]))

        # end‑of‑battle detection -------------------------------------------
        if last_turn == turn:
            break
        last_turn = turn
        if init_att is None:
            init_att, init_def = get_army_strengths(state_json)
        time.sleep(CHECK_INTERVAL)

    # final performance ------------------------------------------------------
    final_state = read_json(BATTLE_JSON_PATH)
    if final_state:
        fin_att, fin_def = get_army_strengths(final_state)
        perf = compute_performance(init_def - fin_def, init_att - fin_att)
        print(f"battle done – performance={perf:.3f}")

    organize_export_files()
    close_vcmi_process()


if __name__ == "__main__":
    battle_loop()
