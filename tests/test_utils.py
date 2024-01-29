from transformers import AutoTokenizer

from src.constants import MODEL
from src.utils import add_token_to_tokenizer


def test_add_token_to_tokenizer(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.save_pretrained(tmp_path)

    new_token = "Ġtestwithnewtoken"
    add_token_to_tokenizer(tmp_path, new_token)

    edited_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    token = edited_tokenizer.tokenize(" testwithnewtoken")[0]
    assert token == new_token


def test_add_token_to_tokenizer_multiple(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.save_pretrained(tmp_path)

    new_tokens = ["Ġtestwithnewtoken", "Ġtestwithnewnewtoken"]
    for new_token in new_tokens:
        add_token_to_tokenizer(tmp_path, new_token)

    edited_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    token = edited_tokenizer.tokenize(" testwithnewtoken")[0]
    assert token == new_tokens[0]

    token = edited_tokenizer.tokenize(" testwithnewnewtoken")[0]
    assert token == new_tokens[1]
