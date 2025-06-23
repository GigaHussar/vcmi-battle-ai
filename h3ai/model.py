import torch
import torch.nn as nn
import torch.nn.functional as F
from battle_state_to_tensor import STATE_DIM    # you can also hard-code 2970
from predictor_helpers    import extract_all_possible_commands
from model                import ActionEncoder  # or adjust import path if needed

# Map your JSON type IDs → string labels (only 0,1,4,5 used)
ACTION_TYPE_MAP = {
    0: "defend",
    1: "wait",
    4: "move",
    5: "melee",
}
# Invert for embedding lookup
ACTION_TYPE_IDS = {v: k for k, v in ACTION_TYPE_MAP.items()}
N_ACTION_TYPES  = len(ACTION_TYPE_MAP)

# Board geometry for normalizing coords
WIDTH_FULL      = 17
WIDTH_PLAYABLE  = 15
HEIGHT          = 11

def hex_to_coords(hex_id):
    """
    Convert a single hex ID into (x,y) on the full grid.
    """
    x = hex_id % WIDTH_FULL
    y = hex_id // WIDTH_FULL
    return x, y

def encode_hex_field(hex_idx):
    """
    Turn a hex index into [x_norm, y_norm, valid_flag].
    - If hex_idx < 0 or None, returns [0,0,0].
    - Otherwise returns [x/WIDTH_PLAYABLE, y/HEIGHT, 1].
    """
    if hex_idx is None or hex_idx < 0:
        return torch.tensor([0.0, 0.0, 0.0])
    x, y = hex_to_coords(hex_idx)
    # clamp to playable area before normalizing
    x = min(max(x, 0), WIDTH_PLAYABLE - 1)
    y = min(max(y, 0), HEIGHT        - 1)
    return torch.tensor([x / (WIDTH_PLAYABLE - 1),
                         y / (HEIGHT        - 1),
                         1.0])  # valid flag = 1

class ActionEncoder(nn.Module):
    """
    Turn each structured action dict into a fixed-length vector φ_a(a).
    - type_emb: converts action type → 8-dim embedding.
    - encode_hex_field: converts hex1/hex2 → 3 floats each.
    Final φ_a(a) size = 8 + 3 + 3 = 14.
    """
    def __init__(self, type_emb_dim: int = 8):
        super().__init__()
        # embedding table for the 4 action types
        self.type_emb = nn.Embedding(N_ACTION_TYPES, type_emb_dim)

    def forward(self, action_dicts: list[dict]) -> torch.Tensor:
        """
        action_dicts: list of length k, each dict has:
           - "type":    str in {"wait","defend","move","melee"}
           - "hex1":    int (or missing) for primary target
           - "hex2":    int (or missing) for secondary target (melee)
        Returns:
           Tensor of shape [k, 14], where each row is
             [ type_embed (8d) ∥ hex1_feat (3d) ∥ hex2_feat (3d) ].
        """
        batch = []
        for a in action_dicts:
            # 1) look up the integer ID of the action type
            t_idx = ACTION_TYPE_IDS[a["type"]]
            # 2) embed into a vector of size type_emb_dim
            e_t   = self.type_emb(torch.tensor(t_idx))

            # 3) encode first hex (move target or melee approach)
            f1 = encode_hex_field(a.get("hex1", -1))
            # 4) encode second hex (only for melee; else sentinel)
            f2 = encode_hex_field(a.get("hex2", -1))

            # 5) concatenate into one φ_a(a) vector of size 14
            batch.append(torch.cat([e_t, f1, f2], dim=0))

        # stack into tensor [k, 14]
        return torch.stack(batch, dim=0)

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

EMBED_DIM = 64   # joint embedding size

class BattleCommandScorer(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_feat_dim=14, embed_dim=EMBED_DIM):
        super().__init__()
        # 1) State projection: STATE_DIM → 256 → 128 → EMBED_DIM
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # 2) Raw action encoder (φₐ): list[dict] → [k, action_feat_dim]
        self.action_enc = ActionEncoder(type_emb_dim=8)

        # 3) Action projection: action_feat_dim → EMBED_DIM
        self.action_proj = nn.Sequential(
            nn.Linear(action_feat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, state_vec: torch.Tensor, action_dicts: list[dict]) -> torch.Tensor:
        """
        Args:
          state_vec:   Tensor [STATE_DIM]
          action_dicts: list of k action‐dicts from extract_all_possible_commands()
        Returns:
          scores:      Tensor [k] of scalar scores
        """
        # Embed the state once
        s_emb = self.state_net(state_vec)           # [EMBED_DIM]

        # Encode & project actions
        a_feats = self.action_enc(action_dicts)      # [k, action_feat_dim]
        a_emb   = self.action_proj(a_feats)          # [k, EMBED_DIM]

        # Score = dot(s_emb, each a_emb[i])
        # expand s_emb to [k, EMBED_DIM]
        k      = a_emb.size(0)
        s_tile = s_emb.unsqueeze(0).expand(k, -1)    # [k, EMBED_DIM]
        scores = (s_tile * a_emb).sum(dim=1)         # [k]

        return scores

# === Test run ===
def test_model():
    model = BattleCommandScorer()
    dummy_input = torch.rand(10, 18)  # 10 commands with same state vector
    output = model(dummy_input)
    print("✅ Output shape:", output.shape)  # should be [10]
    print("🔢 Scores:", output.detach().numpy())

test_model()
