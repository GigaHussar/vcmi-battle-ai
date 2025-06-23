import torch
import numpy as np
import json
from pathlib import Path

from model                import BattleCommandScorer
from predictor_helpers    import extract_all_possible_commands
from battle_state_to_tensor import encode_battle_state_from_json

# ---- CONFIG ----
STATE_FILE    = Path("/Users/syntaxerror/vcmi/export/battle.json")
ACTIONS_FILE  = Path("/Users/syntaxerror/vcmi/export/possible_actions.json")
MODEL_WEIGHTS = Path("model_weights.pth")

# 1) Instantiate your model once, load weights if available
model = BattleCommandScorer()
if MODEL_WEIGHTS.exists():
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.eval()

def predict_best_command():
    try:
        # 2) Load & encode state
        state_json = json.loads(STATE_FILE.read_text())
        feats, c_ids, f_ids = encode_battle_state_from_json(state_json)
        # flatten & concat into one vector of length 2970
        state_vec = torch.from_numpy(
            np.concatenate([feats.flatten(),
                            c_ids.flatten(),
                            f_ids.flatten()])
        ).float()

        # 3) Load & extract actions
        actions_data  = json.loads(ACTIONS_FILE.read_text())
        action_dicts  = extract_all_possible_commands(actions_data)
        if not action_dicts:
            print("⚠️ No legal actions.")
            return None

        # 4) Score actions
        with torch.no_grad():
            scores = model(state_vec, action_dicts)  # [k]

        # 5) Pick best (or sample)
        best_idx = scores.argmax().item()
        chosen  = action_dicts[best_idx]

        print(f"🧠 Model chose: {chosen}")
        return chosen

    except Exception as e:
        print("❌ Predictor failed:", e)
        return None

