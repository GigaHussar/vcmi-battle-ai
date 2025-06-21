import json
import torch
import numpy as np
from model import BattleStateEvaluator
from battle_state_to_tensor import encode_battle_state_from_json

# === Load state from JSON ===
with open("/Users/syntaxerror/vcmi/export/battle.json") as f:
    battle_data = json.load(f)

features, creature_ids, faction_ids = encode_battle_state_from_json(battle_data)

# === Convert to tensors ===
features = torch.tensor(features).unsqueeze(0).float()         # [1, 11, 15, 16]
creature_ids = torch.tensor(creature_ids).unsqueeze(0).long()  # [1, 11, 15]
faction_ids = torch.tensor(faction_ids).unsqueeze(0).long()    # [1, 11, 15]

# === Load model ===
model = BattleStateEvaluator()
model.load_state_dict(torch.load("value_model.pth"))
model.eval()

# === Predict ===
with torch.no_grad():
    prediction = model(features, creature_ids, faction_ids)
    print(f"🔮 Predicted battle value: {prediction.item():.4f}")
