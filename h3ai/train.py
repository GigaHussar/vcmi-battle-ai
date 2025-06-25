import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from model import StateEncoder, ActionEncoder, ActionProjector, STATE_DIM, EMBED_DIM,FEATURE_DIM


from model import CompatibilityScorer, BattleTurnDataset
from paths import MASTER_LOG, EXPORT_DIR, MODEL_WEIGHTS

def collate_fn(batch):
    # batch is a list of dicts: 'state', 'actions', 'chosen_idx'
    states = [item['state'] for item in batch]
    actions = [item['actions'] for item in batch]
    targets = torch.tensor([item['chosen_idx'] for item in batch], dtype=torch.long)

    # Pad sequences to the same length (batch first)
    states_padded = pad_sequence(states, batch_first=True, padding_value=0)
    actions_padded = pad_sequence(actions, batch_first=True, padding_value=0)
    
    return {
        'state': states_padded,
        'actions': actions_padded,
        'chosen_idx': targets,
    }

# Compose the four modules into one training model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_enc     = StateEncoder(STATE_DIM, EMBED_DIM).to(device)
action_enc    = ActionEncoder().to(device)
action_proj   = ActionProjector(FEATURE_DIM, EMBED_DIM).to(device)
compat_scorer = CompatibilityScorer().to(device)

def model(state_vec, action_dicts):
    # forward pass for training
    s_emb = state_enc(state_vec)
    raw_feats = action_enc(action_dicts)
    feats_b = raw_feats.unsqueeze(0).to(device)
    a_emb = action_proj(feats_b)
    return compat_scorer(s_emb, a_emb)

def train(
    log_csv: str,
    data_dir: str,
    output_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = None,
):
    # Device setup
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Dataset and split
    dataset = BattleTurnDataset(log_csv, data_dir)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Model, loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            states = batch['state'].to(device)
            actions = batch['actions'].to(device)
            targets = batch['chosen_idx'].to(device)

            scores = model(states, actions)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * states.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                actions = batch['actions'].to(device)
                targets = batch['chosen_idx'].to(device)
                scores = model(states, actions)
                loss = criterion(scores, targets)
                total_val_loss += loss.item() * states.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BattleCommandScorer on collected data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--device", type=str, default=None, help="Compute device (e.g. 'cpu' or 'cuda')")
    args = parser.parse_args()

    train(
        log_csv=str(MASTER_LOG),
        data_dir=str(EXPORT_DIR),
        output_path=str(MODEL_WEIGHTS),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
    )

