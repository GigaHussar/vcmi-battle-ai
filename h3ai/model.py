import torch
import torch.nn as nn
import torch.nn.functional as F

class BattleCommandScorer(nn.Module):
    def __init__(self, input_size=18, hidden_size=64):
        """
        Model that scores individual battle commands given a game state vector.
        It outputs a single scalar value (logit) per command.
        """
        super(BattleCommandScorer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)  # single score output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)  # shape: [B], one score per input

# === Test run ===
def test_model():
    model = BattleCommandScorer()
    dummy_input = torch.rand(10, 18)  # 10 commands with same state vector
    output = model(dummy_input)
    print("✅ Output shape:", output.shape)  # should be [10]
    print("🔢 Scores:", output.detach().numpy())

test_model()
