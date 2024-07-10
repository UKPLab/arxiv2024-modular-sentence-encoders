import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)


class MultiParallelExample:
    def __init__(self, anchors, positives, negatives=None):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives


class ParaphraseDataset:
    def __init__(self, langs: Iterable[str], dir_path: str):
        """
        Multilingual paraphrase dataset.

        Args:
            langs (Iterable[str]): A list of language codes. 
            dir_path (str): Path of the directory with paraphrase data files. 
        """
        self.langs = langs
        self.data = []  # A list [MultiParallelExmpale, ...]
        self.load_data(dir_path)

    def load_data(self, dir_path):
        # Determine whether the dataset has hard negatives (triplets) or not (pairs)
        if os.path.exists(f"{dir_path}/{self.langs[0]}_3.txt"):
            num_cols = 3
        else:
            num_cols = 2

        multilingual_texts = [[] for _ in range(num_cols)]

        for lang in self.langs:
            for i in range(num_cols):
                with open(f"{dir_path}/{lang}_{i+1}.txt", encoding="utf-8") as f:
                    multilingual_texts[i].append([line.strip() for line in f])

        zipped_multilingual_texts = [
            list(zip(*parallel_text)) for parallel_text in multilingual_texts
        ]

        for cols in zip(*zipped_multilingual_texts):
            example = MultiParallelExample(*cols)
            self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.data[idx]
