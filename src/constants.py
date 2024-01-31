import torch

from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESOURCES_DIR = ROOT / "resources"
STATS_DIR = ROOT / "stats"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = "microsoft/phi-1_5"
