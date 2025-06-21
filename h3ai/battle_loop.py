import os
import json
import socket
import time
import csv
import random
from pathlib import Path
from predictor import predict_best_command
from predictor_helpers import extract_all_possible_commands
from train import train
import runvcmi


# === CONFIGURATION ===
EXPORT_DIR = Path("/Users/syntaxerror/vcmi/export")
ACTIONS_FILE = EXPORT_DIR / "possible_actions.json"
STATE_FILE = EXPORT_DIR / "battle.json"
SOCKET_PORT = 5000
SOCKET_HOST = "localhost"
CHECK_INTERVAL = 1.5
LOG_FILE = EXPORT_DIR / "battle_results.csv"
battle_counter = 0


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

def log_battle_result(
    game_id,
    reward,
    attacker_start,
    attacker_end,
    defender_start,
    defender_end,
    performance
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "game_id",
                "reward",
                "attacker_start",
                "attacker_end",
                "defender_start",
                "defender_end",
                "performance"
            ])

def log_training_example(game_id, state_vector, command, commands):
    # Define the full path to the training CSV
    log_file = Path("/Users/syntaxerror/vcmi/export/training_data.csv")
    
    # Ensure the export folder exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if the file already exists to decide whether to write the header
    file_exists = log_file.exists()

    # Determine which index the chosen command appears at in the available commands list
    # If it's not in the list, set index to -1 (invalid/unknown)
    chosen_index = commands.index(command) if command in commands else -1

    # Convert the list of commands into a single pipe-separated string
    commands_str = "|".join(commands)

    # Open the file in append mode and write the row
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        # If file is new, write the CSV header row
        if not file_exists:
            header = ["game_id"] + [f"f{i}" for i in range(len(state_vector))] + ["chosen_index", "commands"]
            writer.writerow(header)
        
        # Write a new row with: game ID, state features, chosen index, and all commands
        row = [game_id] + list(state_vector) + [chosen_index, commands_str]
        writer.writerow(row)

def compute_performance(kills: float, losses: float) -> float:
    """
    Returns the fraction of total casualties that were enemy kills.
    If there were no casualties, returns 0.0.
    """
    total = kills + losses
    if total <= 0:
        return 0.0
    return kills / total

def battle_loop():
    print("🧠 Agent started. Waiting for battle to begin...")
    last_turn = -1
    initial_attacker_strength = None
    initial_defender_strength = None
    game_id = int(time.time())

    #generate tensors once, at the start of the battle
    os.system(f"python3 battle_state_to_tensor.py {game_id}")

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

        '''
        command = predict_best_command()

        if command:
            print(f"👉 Turn {current_turn}: sending command: {command}")
            send_command(command)
            # Read available actions from JSON
            actions_data = read_json(ACTIONS_FILE)
            commands = extract_all_possible_commands(actions_data)

            # Log full training example including available commands and chosen index
            log_training_example(game_id, state_vector, command, commands)
        else:
            print("❌ No valid commands found.")
            break
        '''

        time.sleep(CHECK_INTERVAL)

    # After battle ends
    state = read_json(STATE_FILE)
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

        # log everything
        log_battle_result(
            game_id,
            initial_attacker_strength,
            final_attacker_strength,
            initial_defender_strength,
            final_defender_strength,
            performance=performance
            )
        print("✅ Battle result logged.")
    else:
        print("⚠️ Could not read final state.")

    # Increment battle counter after each battle
    global battle_counter
    battle_counter += 1  

'''
for i in range(20):
    runvcmi.open_vcmi_process()
    runvcmi.control_vcmi_ui()
    battle_loop()
    #close the game
    runvcmi.close_vcmi_process()
'''