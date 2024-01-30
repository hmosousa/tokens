from transformers import AutoTokenizer, AutoModelForCausalLM

from src.constants import MODEL
from src.utils import add_token_to_tokenizer, compute_perplexity


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


def test_compute_perplexity():
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    text = "This is a test."
    ppl = compute_perplexity(model, tokenizer, text)
    assert ppl > 0


def test_compute_perplexity_batch():
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    text = ["This is a test.", "This is a second test.", "this is a third test."]
    ppl = compute_perplexity(model, tokenizer, text)
    assert ppl > 0
