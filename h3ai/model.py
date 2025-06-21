import torch
import torch.nn as nn
import torch.nn.functional as F

class BattleStateEvaluator(nn.Module):
    # Embedding layers with safe upper bounds based on game engine specs:
    # - Creature IDs assumed < 300
    # - Faction IDs assumed < 15
    def __init__(self, num_creatures=300, num_factions=15):
        super().__init__()

        # === EMBEDDINGS ===
        # Learnable dense representations for categorical unit info
        self.creature_embed = nn.Embedding(num_creatures, 8)  # creature_id → 8D vector
        self.faction_embed = nn.Embedding(num_factions, 4)    # faction_id → 4D vector

        # === CONVOLUTIONAL LAYERS ===
        # Input has 16 raw features + 8 creature emb + 4 faction emb = 28 channels
        self.conv1 = nn.Conv2d(28, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Adaptive average pooling reduces spatial map to [1 x 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fully connected layer maps 64 pooled features → single scalar output
        self.fc = nn.Linear(64, 1)

    def forward(self, features, creature_ids, faction_ids):
        """
        Args:
            features: Tensor [B, 11, 15, 16] - raw battlefield data per tile
            creature_ids: Tensor [B, 11, 15] - creature class ID per tile
            faction_ids: Tensor [B, 11, 15] - faction ID per tile
        Returns:
            output: Tensor [B, 1] - predicted value for this battlefield state
        """

        # === EMBEDDING LOOKUP ===
        # Convert integer IDs into learned feature vectors
        creature_vecs = self.creature_embed(creature_ids)  # [B, 11, 15, 8]
        faction_vecs = self.faction_embed(faction_ids)     # [B, 11, 15, 4]

        # === CONCATENATE ALONG LAST DIMENSION ===
        # Merge raw features and embedded info → [B, 11, 15, 28]
        x = torch.cat([features, creature_vecs, faction_vecs], dim=-1)

        # === PERMUTE FOR CNN ===
        # CNN expects [B, C, H, W], so we move channels (28) to second axis
        x = x.permute(0, 3, 1, 2)  # [B, 28, 11, 15]

        # === CONVOLUTIONAL PROCESSING ===
        # Two layers of convolution + ReLU to extract spatial/tactical patterns
        x = F.relu(self.conv1(x))  # [B, 32, 11, 15]
        x = F.relu(self.conv2(x))  # [B, 64, 11, 15]

        # === GLOBAL POOLING ===
        # Compress spatial map to a flat vector using average pooling
        x = self.pool(x)           # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]

        # === FINAL PREDICTION ===
        # Output a scalar per batch item (e.g. win likelihood or battle value)
        return self.fc(x)          # [B, 1]

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
