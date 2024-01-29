import logging
from pathlib import Path

from transformers import AutoTokenizer

from src.constants import RESOURCES_DIR
from src.utils import add_token_to_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_tokenizers(
    path_original_tokenizer: Path,
    path_trained_tokenizer: Path,
    path_merged_tokenizer: Path,
):
    """Add the tokens from tokenizer1 path_trained_to tokenizer2."""

    original_tokenizer = AutoTokenizer.from_pretrained(path_original_tokenizer)
    trained_tokenizer = AutoTokenizer.from_pretrained(path_trained_tokenizer)

    # Start by initializing the merged tokenizer as the original tokenizer.
    original_tokenizer.save_pretrained(path_merged_tokenizer)

    new_tkns = list(set(trained_tokenizer.vocab.keys()) - set(original_tokenizer.vocab.keys()))

    for tkn in new_tkns:
        logger.info(f"Adding token {tkn} to the tokenizer.")
        add_token_to_tokenizer(path_merged_tokenizer, tkn)

    merged_tokenizer = AutoTokenizer.from_pretrained(path_merged_tokenizer)

    return merged_tokenizer


path_original_tokenizer = RESOURCES_DIR / "original_tokenizer"
path_trained_tokenizer = RESOURCES_DIR / "trained_tokenizer"
path_merged_tokenizer = RESOURCES_DIR / "merged_tokenizer"

merged_tokenizer = merge_tokenizers(path_original_tokenizer, path_trained_tokenizer, path_merged_tokenizer)
