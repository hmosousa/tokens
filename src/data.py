from nltk.tokenize import sent_tokenize, word_tokenize

from src.constants import DATA_DIR


def transform_data(data):
    sentences = []
    with open(DATA_DIR / "train.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                sents = sent_tokenize(line)
                for sent in sents:
                    words = word_tokenize(sent)
                    if len(words) > 5:
                        sentences.append(sent)
        sentences = list(set(sentences))
    return sentences


def read_data():
    with open(DATA_DIR / "transformed.txt", "r") as f:
        data = f.readlines()
    return data
