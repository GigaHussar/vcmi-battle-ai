"""
Battle‑playing agent:
   • encodes current battle state + every legal action
   • feeds them through trained network
   • selects action with highest predicted performance
   • sends it back to VCMI
"""
import time
from using_model_api import query_gemma3_with_battle_state
from _helpers_do_not_touch import read_json, send_command, get_army_strengths
from _paths_do_not_touch import (
    MODEL_WEIGHTS, EXPORT_DIR, BATTLE_JSON_PATH, ACTIONS_FILE
)
from _runvcmi_do_not_touch import open_vcmi_process, close_vcmi_process

CHECK_INTERVAL = 4


def battle_loop():

    open_vcmi_process()
    time.sleep(5)
    send_command("open_load_menu")
    time.sleep(1)
    send_command("lobby_accept")
    time.sleep(2)
    send_command("move_active_hero_right")
    time.sleep(2)

    game_id = int(time.time())
    turn_number = None
    last_turn = None
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

        time.sleep(2)
        # end‑of‑battle detection -------------------------------------------
        # Use already parsed actions_json
        turn_number = actions_json.get("turn")
        print(turn_number)

        print(last_turn)
        if last_turn == turn_number:
            break
        last_turn = turn_number
        if init_att is None:
            init_att, init_def = get_army_strengths(state_json)
        time.sleep(CHECK_INTERVAL)
  
    close_vcmi_process()


battle_loop()
