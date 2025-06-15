import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import BattleCommandScorer

# === CONFIGURATION ===
TRAIN_DATA_FILE = "logs/training_data.csv"
REWARD_DATA_FILE = "logs/battle_log.csv"
MODEL_SAVE_PATH = "model_weights.pth"
INPUT_DIM = 18
EPOCHS = 5
BATCH_SIZE = 1  # Train one battle at a time for full context
LEARNING_RATE = 1e-3

def load_training_data():
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    reward_df = pd.read_csv(REWARD_DATA_FILE)
    merged = train_df.merge(reward_df[["game_id", "reward"]], on="game_id")
    return merged

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
            xb = torch.tensor(state_batch)
            yb = torch.tensor([chosen_index])
            reward = torch.tensor([row["reward"]], dtype=torch.float32)

            logits = model(xb).unsqueeze(0)  # shape [1, N]
            loss = loss_fn(logits, yb) * reward  # scale by reward

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