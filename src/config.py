import os

RANDOM_STATE = 42
TEST_SIZE = 0.2

IMAGE_SIZE = (32, 32)
NUM_CHANNELS = 3

DATA_DIR = "data"
FIGURES_DIR = os.path.join("report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
