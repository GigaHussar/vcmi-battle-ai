from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
EXPORT_DIR = BASE_PATH / "export"
H3AI_DIR = BASE_PATH / "h3ai"
BATTLE_JSON_PATH = EXPORT_DIR / "battle.json"
ACTIONS_FILE = EXPORT_DIR / "possible_actions.json"
MODEL_WEIGHTS = H3AI_DIR / "model_weights.pth"
MASTER_LOG = EXPORT_DIR / "master_log.csv"