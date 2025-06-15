import torch
import numpy as np
import json
from pathlib import Path
from battle_state_encoder import encode_battle_state
from model import BattlePolicyNet

# === CONFIGURATION ===
STATE_FILE = Path("/Users/syntaxerror/vcmi/export/battle.json")
ACTIONS_FILE = Path("/Users/syntaxerror/vcmi/export/possible_actions.json")
ACTION_TYPE_MAP = {
    4: "move",
    5: "melee",
    6: "melee",
#    2: "shoot",  # placeholder if needed later
    1: "wait",
    0: "defend"
}
ACTION_INDEX_MAP = {
    "move": 0,
    "melee": 1,
#    "shoot": 2,
    "wait": 3,
    "defend": 4
}
REVERSE_ACTION_INDEX = {v: k for k, v in ACTION_INDEX_MAP.items()}

# === PREDICTION FUNCTION ===

def predict_best_action_type():
    try:
        # Load and encode state
        with open(STATE_FILE) as f:
            state_json = json.load(f)
        features = encode_battle_state(state_json)
        input_tensor = torch.tensor(features).unsqueeze(0)

        # Load model
        model = BattlePolicyNet()
        if Path("model_weights.pth").exists():
            model.load_state_dict(torch.load("model_weights.pth"))
        model.eval()

        # Predict action probabilities
        with torch.no_grad():
            probs = model(input_tensor).numpy()[0]

        # Load available actions
        with open(ACTIONS_FILE) as f:
            actions_data = json.load(f)

        available_types = set()
        for action in actions_data.get("actions", []):
            type_id = action.get("type")
            action_name = ACTION_TYPE_MAP.get(type_id)
            if action_name:
                available_types.add(action_name)

        if not available_types:
            print("⚠️ No available action types.")
            return None

        # Filter probabilities for legal actions only
        legal_indices = [ACTION_INDEX_MAP[a] for a in available_types]
        legal_probs = probs[legal_indices]
        legal_probs /= legal_probs.sum()

        # Sample from legal actions
        chosen_index = np.random.choice(legal_indices, p=legal_probs)
        chosen_action = REVERSE_ACTION_INDEX[chosen_index]
        print(f"🧠 Model chose action type: {chosen_action}")
        return chosen_action

    except Exception as e:
        print("❌ Predictor failed:", e)
        return None

# === TEST RUN ===
if __name__ == "__main__":
    predict_best_action_type()
