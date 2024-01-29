from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import compute_perplexity
from src.data import read_data
from src.constants import MODEL

data = read_data()

model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

ppls = [compute_perplexity(model, tokenizer, entry) for entry in data[:10]]
print(f"Perplexity: {ppls}")
