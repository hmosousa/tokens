import multiprocessing as mp
from pathlib import Path

from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from src.constants import DATA_DIR


def transform_data(filepath: Path) -> list[str]:
    sentences = []
    with open(filepath, "r") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line != "":
                sents = sent_tokenize(line)
                sentences += sents
        sentences = list(set(sentences))
    return sentences


def main():
    input_filepath = DATA_DIR / "train.txt"
    sentences = transform_data(input_filepath)

    output_filepath = DATA_DIR / "transformed.txt"
    output_filepath.write_text("\n".join(sentences))


if __name__ == "__main__":
    main()
