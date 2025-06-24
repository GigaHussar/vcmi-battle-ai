# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import BattleCommandScorer, BattleTurnDataset

# Config
LOG_CSV  = "export/master_log.csv"
DATA_DIR = "export"
BATCH    = 32
LR       = 1e-4
EPOCHS   = 5
SAVE_TO  = "export/model_weights.pth"

def collate_fn(batch):
    # batch is list of dicts with state, actions, chosen_idx, reward
    states = torch.stack([b["state"] for b in batch])           # [B, S]
    rewards = torch.stack([b["reward"] for b in batch]).unsqueeze(1)  # [B,1]
    return states, batch  # pass entire batch list so we can handle variable k

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds  = BattleTurnDataset(LOG_CSV, DATA_DIR)
    dl  = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    model = BattleCommandScorer().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        for states, batch in dl:
            states = states.to(device)
            rewards = torch.stack([b["reward"] for b in batch]).unsqueeze(1).to(device)
            # for each example, score its k actions and pick the chosen one
            preds = []
            for i, b in enumerate(batch):
                acts = b["actions"].to(device)                 # [k, F]
                idx  = b["chosen_idx"].item()
                # scorer may expect batched states+actions:
                scores = model(states[i].unsqueeze(0), [acts])[0]  # [k]
                preds.append(scores[idx].unsqueeze(0))
            preds = torch.cat(preds).unsqueeze(1)              # [B,1]

            loss = F.mse_loss(preds, rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * states.size(0)

        avg = total_loss / len(ds)
        print(f"Epoch {epoch}/{EPOCHS} — avg loss: {avg:.4f}")

    torch.save(model.state_dict(), SAVE_TO)
    print(f"Model saved to {SAVE_TO}")

if __name__ == "__main__":
    train()
