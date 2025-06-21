import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import BattleStateEvaluator

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "..", "export", "battle_results.csv")
TENSOR_DIR = os.path.join(SCRIPT_DIR, "..", "h3ai")
MODEL_SAVE_PATH = "value_model.pth"
EPOCHS = 5
LEARNING_RATE = 1e-3

def load_tensor_by_game_id(game_id):
    try:
        features = np.load(os.path.join(TENSOR_DIR, f"battlefield_tensor_{game_id}.npy"))
        creature_ids = np.load(os.path.join(TENSOR_DIR, f"creature_id_tensor_{game_id}.npy"))
        faction_ids = np.load(os.path.join(TENSOR_DIR, f"faction_id_tensor_{game_id}.npy"))
        return features, creature_ids, faction_ids
    except Exception as e:
        print(f"❌ Failed to load tensors for game {game_id}: {e}")
        return None, None, None

def train():
    df = pd.read_csv(RESULTS_CSV)
    model = BattleStateEvaluator()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        count = 0

        for _, row in df.iterrows():
            game_id = row["game_id"]
            performance = float(row["performance"])  # label

            features, creature_ids, faction_ids = load_tensor_by_game_id(game_id)
            if features is None:
                continue

            # Add batch dim and convert to tensors
            xb = torch.tensor(features).unsqueeze(0).float()
            cid = torch.tensor(creature_ids).unsqueeze(0).long()
            fid = torch.tensor(faction_ids).unsqueeze(0).long()
            target = torch.tensor([[performance]], dtype=torch.float32)  # shape [1, 1]

            # Forward + Loss + Backward
            pred = model(xb, cid, fid)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"📚 Epoch {epoch+1}/{EPOCHS} - Avg MSE Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Value model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
