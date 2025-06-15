import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import BattlePolicyNet

# === CONFIGURATION ===
TRAIN_DATA_FILE = "logs/training_data.csv"
REWARD_DATA_FILE = "logs/battle_log.csv"
MODEL_SAVE_PATH = "model_weights.pth"
NUM_ACTIONS = 5
INPUT_DIM = 18
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# === LOAD AND MERGE DATA ===

def load_training_data():
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    reward_df = pd.read_csv(REWARD_DATA_FILE)
    merged = train_df.merge(reward_df[["game_id", "reward"]], on="game_id")
    X = merged[[f"f{i}" for i in range(INPUT_DIM)]].values.astype(np.float32)
    y = merged["action"].values.astype(np.int64)
    rewards = merged["reward"].values.astype(np.float32)
    return X, y, rewards

# === TRAINING FUNCTION ===

def train():
    X, y, rewards = load_training_data()

    model = BattlePolicyNet(input_size=INPUT_DIM, num_actions=NUM_ACTIONS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(reduction="none")  # per-sample loss

    model.train()
    for epoch in range(EPOCHS):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0.0

        for start in range(0, len(X), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = indices[start:end]

            xb = torch.tensor(X[batch_idx])
            yb = torch.tensor(y[batch_idx])
            rb = torch.tensor(rewards[batch_idx])

            logits = model(xb)
            loss_per_sample = loss_fn(logits, yb)
            weighted_loss = (loss_per_sample * rb).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item() * len(batch_idx)

        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# === RUN ===
if __name__ == "__main__":
    train()