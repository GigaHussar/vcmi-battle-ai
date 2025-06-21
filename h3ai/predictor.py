import torch
import numpy as np
import json
from pathlib import Path
from model import BattleCommandScorer
from predictor_helpers import extract_all_possible_commands  # assuming moved to predictor_helpers.py
from battle_state_to_tensor import encode_battle_state_from_json 
# === CONFIGURATION ===
STATE_FILE = Path("/Users/syntaxerror/vcmi/export/battle.json")
ACTIONS_FILE = Path("/Users/syntaxerror/vcmi/export/possible_actions.json")
MODEL_WEIGHTS = "model_weights.pth"

def predict_best_command():
    try:
        with open(STATE_FILE) as f:
            state_json = json.load(f)
        features = encode_battle_state_from_json(state_json)
        input_tensor = torch.tensor(features).unsqueeze(0).float()

        with open(ACTIONS_FILE) as f:
            actions_data = json.load(f)

        commands = extract_all_possible_commands(actions_data)
        if not commands:
            print("⚠️ No executable commands available.")
            return None

        model = BattleCommandScorer()
        if Path(MODEL_WEIGHTS).exists():
            model.load_state_dict(torch.load(MODEL_WEIGHTS))
        model.eval()

        # Map commands to indices
        num_commands = len(commands)
        command_logits = model(input_tensor.repeat(num_commands, 1))  # shape: [N, num_actions]
        scores = command_logits

        probs = torch.softmax(scores, dim=0).detach().numpy()
        chosen_index = np.random.choice(num_commands, p=probs)
        chosen_command = commands[chosen_index]

        print(f"🧠 Model chose command: {chosen_command}")
        return chosen_command

    except Exception as e:
        print("❌ Predictor failed:", e)
        return None
