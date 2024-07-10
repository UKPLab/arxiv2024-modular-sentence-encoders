import torch
from torch import Tensor
from typing import Iterable, Dict

from sentence_transformers import util
from sentence_transformers.losses import MultipleNegativesRankingLoss

from ModularSentenceTransformer import ModularSentenceTransformer

device = torch.device("cuda")


class CustomMNRL(MultipleNegativesRankingLoss):
    """
    Multiple Negatives Ranking Loss for training on single model.
    Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
    """

    def __init__(
        self,
        model: ModularSentenceTransformer,
        batch_size,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
        sonar=False,
    ):
        """
        Args:
            model (SentenceTransformer): a ModularSentenceTransformer model.
            batch_size (_type_): Train batch size.
            scale (float, optional): Output of similarity function is multiplied by scale. Defaults to 20.0.
            similarity_fct (_type_, optional): similarity function between sentence embeddings. Defaults to util.cos_sim.
            sonar (bool, optional): Whether the model is a SONAR model. Defaults to False.
        """
        super().__init__(model, scale=scale, similarity_fct=similarity_fct)
        self.batch_size = batch_size
        self.sonar = sonar

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels):
        if not self.sonar:
            return super().forward(sentence_features, labels)
        else:
            return self.forward_by_lang(sentence_features, labels)

    def embed_sentence(self, sentence_features):
        return [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

    def forward_by_lang(
        self, sentence_features: Iterable[Dict[str, Tensor]], labels: list
    ):
        """
        Embed inputs grouped by language and reassemble them into a batch.

        Args:
            sentence_features (Iterable[Dict[str, Tensor]]): A dictionary {lang: tokenized inputs}.
            labels (list): A list of indices for the tokenized inputs in the sentence_features.

        Returns:
            Loss value
        """
        embs = self.embed_sentence(sentence_features)
        emb_dim = embs[0].shape[1]

        col_set = set([ind_pair[1] for indices in labels for ind_pair in indices])
        num_col = len(col_set)

        # Reassemble the cross-lingual batch by mixing embeddings from each language
        reps = [
            torch.empty(self.batch_size, emb_dim).to(device) for _ in range(num_col)
        ]
        for rep, ind in zip(embs, labels):  # loop over langs
            for i in range(len(ind)):
                bid, col = ind[i]
                reps[col][bid] = rep[i]

        # original MNRL implementation
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)


class CrossModelMNRL(CustomMNRL):
    def __init__(
        self,
        model: ModularSentenceTransformer,
        pivot_model: ModularSentenceTransformer,
        batch_size,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
    ):
        super().__init__(
            model, scale=scale, similarity_fct=similarity_fct, batch_size=batch_size
        )
        self.pivot_model = pivot_model
        self.pivot_model.eval()

    def embed_sentence(self, sentence_features):
        """Embed texts for each language"""
        embs = [
            self.pivot_model(sentence_features[0])["sentence_embedding"],
            self.model(sentence_features[1])["sentence_embedding"],
        ]
        return embs

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels):
        return super().forward_by_lang(sentence_features, labels)
