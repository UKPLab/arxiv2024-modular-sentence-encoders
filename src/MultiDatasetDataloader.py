import random

from sentence_transformers import InputExample


class MultiDatasetDataLoader:
    def __init__(self, datasets: list, batch_size: int, num_langs: int, batch_type: str):
        """
        Dataloder for multiple multilingual paraphrase datasets.
        
        Args:
            datasets (list): A list of multilingual paraphrase datasets. 
            batch_size (int): Batch size.
            num_langs (int): Number of languages. 
            batch_type (str): Either "mono" (monolingual batch) or "cross" (cross-lingual batch).
        """
        
        self.batch_size = batch_size
        self.num_langs = num_langs
        self.batch_type = batch_type

        self.collate_fn = None

        self.dataset_lengths_sum = sum(list(map(len, datasets)))
        self.dataset_idx = list(range(len(datasets)))
        self.dataset_idx_pointer = 0

        random.shuffle(self.dataset_idx)

        self.datasets = []
        for dataset in datasets:
            random.shuffle(dataset)
            self.datasets.append(
                {
                    "elements": dataset,
                    "pointer": 0,
                }
            )

    def __iter__(self):
        for _ in range(int(self.__len__())):
            # Select dataset
            if self.dataset_idx_pointer >= len(self.dataset_idx):
                self.dataset_idx_pointer = 0
                random.shuffle(self.dataset_idx)

            dataset_idx = self.dataset_idx[self.dataset_idx_pointer]
            self.dataset_idx_pointer += 1

            dataset = self.datasets[dataset_idx]
            batch_size = self.batch_size

            if dataset["elements"][0].negatives is not None:
                create_instance = self.create_triplet_instance
            else:
                create_instance = self.create_pair_instance

            # Select languages for the current batch
            langs = self.select_batch_lang()

            # Create batch from the selected dataset
            batch = []
            while len(batch) < batch_size:
                # Select a multi-parallel pair/triplet
                example = dataset["elements"][dataset["pointer"]]
                # Create the monolingual pair/triplet
                example = create_instance(example, langs)
                batch.append(example)

                dataset["pointer"] += 1
                if dataset["pointer"] >= len(dataset["elements"]):
                    dataset["pointer"] = 0
                    random.shuffle(dataset["elements"])

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def select_batch_lang(self, max_num=6):
        """
        Sample a list of languages for creating a batch.

        Args:
            max_num (int, optional): Maximum number of languages in a batch. Defaults to 6.

        Returns:
            list: A list of language indices.
        """
        if self.batch_type == "mono":
            return [random.randint(0, self.num_langs - 1)]
        elif self.batch_type == "cross":
            return random.sample(range(self.num_langs), min(max_num, self.num_langs)) 

    def select_instance_lang(self, batch_lang: int, num_col: int):
        """
        Sample language for each sentence in the pair or triplet

        Args:
            batch_lang (int): Index of the batch language.
            num_col (int): Either 2 (pair) or 3 (triplet).

        Returns:
            list: A list of language indices.
        """
        if self.batch_type == "mono":
            return [batch_lang[0]] * num_col
        elif self.batch_type == "cross":
            if len(batch_lang) > 2:
                return random.sample(batch_lang, num_col)
            else:
                lang1, lang2 = random.sample(batch_lang, 2)
                if num_col > 2:
                    lang3 = lang1 if random.random() < 0.5 else lang2
                    return [lang1, lang2, lang3]
                return [lang1, lang2]

    def create_triplet_instance(self, example, langs):
        lang1, lang2, lang3 = self.select_instance_lang(langs, 3)
        texts = [
            example.anchors[lang1],
            example.positives[lang2],
            example.negatives[lang3],
        ]
        return InputExample(texts=texts, label=[lang1, lang2, lang3])

    def create_pair_instance(self, example, langs):
        lang1, lang2 = self.select_instance_lang(langs, 2)
        texts = [example.anchors[lang1], example.positives[lang2]]
        return InputExample(texts=texts, label=[lang1, lang2])

    def __len__(self):
        return int(self.dataset_lengths_sum / self.batch_size)
