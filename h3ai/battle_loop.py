import os
import json
import socket
import time
import csv
import random
from pathlib import Path
from predictor import predict_best_action_type
from battle_state_encoder import encode_battle_state


# === CONFIGURATION ===
EXPORT_DIR = Path("/Users/syntaxerror/vcmi/export")
ACTIONS_FILE = EXPORT_DIR / "possible_actions.json"
STATE_FILE = EXPORT_DIR / "battle.json"
SOCKET_PORT = 5000
SOCKET_HOST = "localhost"
CHECK_INTERVAL = 1.5
LOG_FILE = EXPORT_DIR / "battle_log.csv"

def choose_first_valid_command_of_type(action_type, actions_data):
    """
    From available actions in actions_data, pick the first valid command
    matching the action_type predicted by the model.
    """
    commands = []

    for action in actions_data.get("actions", []):
        type_id = action.get("type")

        # Move
        if action_type == "move" and type_id == 4:
            for tile in action.get("reachable_tiles", []):
                commands.append(f"move {tile['hex']}")

        # Melee
        elif action_type == "melee" and type_id in (5, 6):
            for target in action.get("melee_targets", []):
                if target.get("can_melee_attack", False):
                    target_hex = target["hex"]
                    from_hex = target["attack_from"]["hex"]
                    commands.append(f"melee {target_hex} {from_hex}")

        # Wait
        elif action_type == "wait" and type_id == 1:
            commands.append("wait")

        # Defend
        elif action_type == "defend" and type_id == 0:
            commands.append("defend")

    if commands:
        return commands[0]  # Choose the first one
    return None  # Fallback if no matching command found

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

def extract_valid_commands(actions_data):
    commands = []

    for action in actions_data.get("actions", []):
        action_type = action.get("type")

        # Defend
        if action_type == 0:
            commands.append("defend")

        # Wait
        elif action_type == 1:
            commands.append("wait")

        # Move
        elif action_type == 4:
            for tile in action.get("reachable_tiles", []):
                commands.append(f"move {tile['hex']}")

        # Melee
        elif action_type in (5, 6):
            for target in action.get("melee_targets", []):
                if target.get("can_melee_attack", False):
                    target_hex = target["hex"]
                    from_hex = target["attack_from"]["hex"]
                    commands.append(f"melee {target_hex} {from_hex}")
        # Ranged

    return commands

def choose_random_action(commands):
    if commands:
        return random.choice(commands)
    return None

def get_army_strengths(state):
    return (
        state.get("army_strength_attacker", 0),
        state.get("army_strength_defender", 0)
    )

def log_battle_result(game_id, reward, atk_start, atk_end, def_end):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["game_id", "reward", "attacker_start", "attacker_end", "defender_end"])
        writer.writerow([game_id, reward, atk_start, atk_end, def_end])

def log_training_example(game_id, state_vector, action_type):
    """
    Appends a training sample to logs/training_data.csv:
    - game_id: unique battle identifier
    - state_vector: encoded state from battle_state_encoder
    - action_type: string like "move", "melee", etc.
    """
    log_file = Path("/Users/syntaxerror/vcmi/export/training_data.csv")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists()

    # Convert action name to numeric index
    action_map = {"move": 0, "melee": 1, "shoot": 2, "wait": 3, "defend": 4}
    action_index = action_map.get(action_type, -1)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["game_id"] + [f"f{i}" for i in range(len(state_vector))] + ["action"]
            writer.writerow(header)
        row = [game_id] + list(state_vector) + [action_index]
        writer.writerow(row)


def battle_loop():
    print("🧠 Agent started. Waiting for battle to begin...")
    last_turn = -1
    initial_attacker_strength = None
    initial_defender_strength = None
    game_id = int(time.time())

    while True:
        state = read_json(STATE_FILE)
        actions_data = read_json(ACTIONS_FILE)

        if not state or not actions_data:
            print("⏳ Waiting for battle state...")
            time.sleep(CHECK_INTERVAL)
            continue

        current_turn = actions_data.get("turn", -1)
        if last_turn == current_turn:
            print("🏁 Battle ended. Collecting result...")
            break

        last_turn = current_turn

        if initial_attacker_strength is None:
            initial_attacker_strength, initial_defender_strength = get_army_strengths(state)

        predicted_type = predict_best_action_type()

        # Encode current state
        state_vector = encode_battle_state(state)
        log_training_example(game_id, state_vector, predicted_type)

        command = choose_first_valid_command_of_type(predicted_type, actions_data)

        if command:
            print(f"👉 Turn {current_turn}: sending command: {command}")
            send_command(command)
        else:
            print("❌ No valid commands found.")
            break

        time.sleep(CHECK_INTERVAL)

    # After battle ends
    state = read_json(STATE_FILE)
    if state:
        final_attacker_strength, final_defender_strength = get_army_strengths(state)
        reward = (
            final_attacker_strength
            - (initial_attacker_strength - final_attacker_strength)
            + (initial_defender_strength - final_defender_strength) * 0.5
        )
        print("📊 Final Result:")
        print(f"   🛡  Attacker: {initial_attacker_strength} → {final_attacker_strength}")
        print(f"   ⚔️  Defender: {initial_defender_strength} → {final_defender_strength}")
        print(f"   ✅ Reward: {reward:.2f}")

        log_battle_result(game_id, reward, initial_attacker_strength, final_attacker_strength, final_defender_strength)
        print("✅ Battle result logged.")
    else:
        print("⚠️ Could not read final state.")

battle_loop()
