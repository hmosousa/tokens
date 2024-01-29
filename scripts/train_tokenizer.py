import logging
from transformers import AutoTokenizer

from src.constants import DATA_DIR, MODEL, RESOURCES_DIR
from src.utils import train_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train a tokenizer on the training data."""
    corpus = (DATA_DIR / "train.txt").read_text()
    lines = corpus.split("\n")
    lines = list(set(lines))

    original_tokenizer = AutoTokenizer.from_pretrained(MODEL)
    original_tokenizer.save_pretrained(RESOURCES_DIR / "original_tokenizer")

    trained_tokenizer = train_tokenizer(original_tokenizer, lines, vocab_size=10_000)

    trained_tokenizer.save_pretrained(RESOURCES_DIR / "trained_tokenizer")


if __name__ == "__main__":
    main()
