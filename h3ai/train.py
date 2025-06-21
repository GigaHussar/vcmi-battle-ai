import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import BattleCommandScorer

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_FILE = os.path.join(SCRIPT_DIR, "..", "export", "training_data.csv")
MODEL_SAVE_PATH = "model_weights.pth"
INPUT_DIM = 18
EPOCHS = 5
BATCH_SIZE = 1  # One action per sample
LEARNING_RATE = 1e-3

# === REWARD SCALING PARAMETERS ===
WEIGHT_ENEMY_DAMAGE = 1.0      # Encourage doing damage
WEIGHT_SELF_LOSS = 1.0         # Discourage taking damage
WEIGHT_TRADE_EFFICIENCY = 0.5  # Encourage positive damage-to-loss ratio

def compute_shaped_reward(row):
    # Input fields expected in training_data.csv:
    # - enemy_damage: amount of HP or unit value dealt to enemies
    # - self_damage: amount of HP or unit value lost by acting unit
    # - reward_base: optional base game-level reward (e.g. win=1, lose=0)

    # Encourage hurting enemies
    r_damage = WEIGHT_ENEMY_DAMAGE * row["enemy_damage"]

    # Penalize own losses
    r_self = -WEIGHT_SELF_LOSS * row["self_damage"]

    # Reward favorable trades (if damage > self-damage, this is positive)
    trade_score = row["enemy_damage"] - row["self_damage"]
    r_trade = WEIGHT_TRADE_EFFICIENCY * trade_score

    # Optional: include long-term base reward
    base = row.get("reward_base", 0.0)

    shaped_reward = r_damage + r_self + r_trade + base
    return shaped_reward

def load_training_data():
    df = pd.read_csv(TRAIN_DATA_FILE)
    df["shaped_reward"] = df.apply(compute_shaped_reward, axis=1)
    return df

def train():
    data = load_training_data()
    model = BattleCommandScorer(input_size=INPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
        total_loss = 0.0

        for _, row in data.iterrows():
            state_vec = np.array([row[f"f{i}"] for i in range(INPUT_DIM)], dtype=np.float32)
            chosen_index = int(row["chosen_index"])
            commands = row["commands"].split("|")
            num_cmds = len(commands)

            state_batch = np.tile(state_vec, (num_cmds, 1))  # shape [N, 18]
            xb = torch.tensor(state_batch, dtype=torch.float32)
            yb = torch.tensor([chosen_index])
            reward = torch.tensor([row["shaped_reward"]], dtype=torch.float32)

            logits = model(xb).unsqueeze(0)  # [1, N]
            loss = loss_fn(logits, yb) * reward  # scale loss by reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        print(f"📚 Epoch {epoch+1}/{EPOCHS} - Avg Weighted Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()