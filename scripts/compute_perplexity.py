import json
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import DEVICE, MODEL, RESOURCES_DIR, STATS_DIR
from src.data import read_data
from src.utils import batch_iterator, compute_perplexity

TOKENIZER = "original_tokenizer"
BATCH_SIZE = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Using device {DEVICE}")

logger.info("Loading data.")
data = read_data()

logger.info(f"Loading model {MODEL} and tokenizer {TOKENIZER}.")
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(RESOURCES_DIR / TOKENIZER, device_map="auto")

logger.info("Computing perplexity.")
n_batches = len(data) // BATCH_SIZE
data_iter = batch_iterator(data, batch_size=BATCH_SIZE)
ppls = []
for batch in tqdm(data_iter, total=n_batches):
    ppls.append(compute_perplexity(model, tokenizer, batch))
ppls = torch.tensor(ppls)


results = {
    "model": MODEL,
    "tokenizer": TOKENIZER,
    "perplexity": {
        "mean": ppls.mean().item(),
        "min": ppls.min().item(),
        "max": ppls.max().item(),
    },
}

STATS_DIR.mkdir(exist_ok=True)

runid = datetime.now().strftime("%Y%m%d%H%M%S")
output_filepath = STATS_DIR / f"perplexity_{runid}.jsonl"
json.dump(results, output_filepath.open("w"), indent=4)
