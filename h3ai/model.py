import torch
import torch.nn as nn
import torch.nn.functional as F

class BattlePolicyNet(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_actions=5):
        """
        input_size: number of features from the encoder
        num_actions: number of action types (e.g., move, melee, wait, defend, shoot)
        """
        super(BattlePolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)

# === Test run ===
def test_model():
    model = BattlePolicyNet()
    dummy_input = torch.rand(1, 18)  # batch of one state vector
    output = model(dummy_input)
    print("✅ Model output shape:", output.shape)
    print("🔢 Action probabilities:", output.detach().numpy())

test_model()