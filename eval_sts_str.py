import logging

import pandas as pd

from sentence_transformers import LoggingHandler

from sklearn.metrics.pairwise import paired_cosine_distances

from scipy.stats import spearmanr

from src.ModularSentenceTransformer import ModularSentenceTransformer

from eval_utils import load_adapter_model
from eval_data_utils import load_additional_sts17_data, load_sts17_data, load_str24_data, load_sts22_data, load_kardes_data


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


def eval_sts_str(langs: list, dataset: dict, model: ModularSentenceTransformer, model2: ModularSentenceTransformer = None, sonar: bool = False):
    """
    Run STS or STR evaluation on a language pair. 

    Args:
        langs (list): The language pair: [lang1, lang2]
        dataset (dict): Dataset of the language pair. {'sentences1: [...], 'sentences2: [...], 'scores': [...]}
        model (ModularSentenceTransformer): Model to encode the sentences. If model2 is None, also encode the sentences2.
        model2 (ModularSentenceTransformer, optional): Model to encode the second sentences (sentences2). 
        sonar (bool, optional): Whether the model is a SONAR model. Defaults to False.

    Returns:
        Spearman correlation between cosine similarity and gold labels. 
    """

    sents1 = dataset['sentences1']
    sents2 = dataset['sentences2']
    labels = dataset['scores']
    lang1, lang2 = langs

    embs1 = model.encode(sents1, lang=lang1 if sonar else None, show_progress_bar=False)

    sents2_encoder = model2 if model2 else model
    embs2 = sents2_encoder.encode(sents2, lang=lang2 if sonar else None, show_progress_bar=False)

    cosine_scores = 1 - (paired_cosine_distances(embs1, embs2))
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    return eval_spearman_cosine


## Load evaluation datasets
logger.info("Load evaluation datasets")

sts17 = load_sts17_data()
sts17_additional = load_additional_sts17_data()
sts17.update(sts17_additional)

sts22 = load_sts22_data()
str24 = load_str24_data()
kardes_sts = load_kardes_data()

data = {
    'extended_sts17': sts17,
    'kardes_sts': kardes_sts,
    'str24': str24,
    'sts22': sts22,
}


## Evaluate single-model baselines: 
logger.info("Evaluate singe model")
model = ModularSentenceTransformer('sentence-transformers/LaBSE')  # an example
model.max_seq_length = 512

for dataset_name, datasets in data.items():
    results = dict()
    for lang_pair, dataset in datasets.items():
        score = eval_sts_str(lang_pair, dataset, model=model, sonar=False)
        lang1, lang2 = lang_pair
        rev_pair = '-'.join([lang2, lang1])
        if rev_pair in results:
            results[rev_pair] = (score + results[rev_pair]) / 2  # average of lang1-lang2 and lang2-lang1 results
        else:
            results['-'.join(lang_pair)] = score
    df = pd.DataFrame(
        list(results.items()),
        columns=["language", "spearman cosine"],
    )
    df.to_csv(f'{dataset_name}_baseline_results.csv')



## Evaluate modular models
def load_lang_model(lang):
    """
    Load the monolingual sentence encoder. 
    Assume monolingual models are stored with the same name format. 
    """
    return load_adapter_model(
        model_path=f'model/mono_encoder/labse_{lang}',
        adapter_path=f'model/cla_adapter/labse_{lang}_adapter' if lang!='eng' else None
    )

logger.info("Evaluate modular models")

for dataset_name, datasets in data.items():
    results = dict()
    for lang_pair, dataset in datasets.items():
        lang1, lang2 = lang_pair

        if lang1 == lang2:  # Monolingual evaluation
            model = load_lang_model(lang1)
            score = eval_sts_str(lang_pair, dataset, model=model, sonar=False)
            results['-'.join(lang_pair)] = score

        else:  # Cross-lingual evaluation
            model1 = load_lang_model(lang1)
            model2 = load_lang_model(lang2)

            # Activate cross-lingual adapters
            if lang1 != 'eng':
                model1[0].auto_model.set_active_adapters('cla')
            if lang2 != 'eng':
                model2[0].auto_model.set_active_adapters('cla')

            score = eval_sts_str(lang_pair, dataset, model=model1, model2=model2, sonar=False)
            rev_pair = '-'.join([lang2, lang1])
            if rev_pair in results:
                results[rev_pair] = (score + results[rev_pair]) / 2  # average of lang1-lang2 and lang2-lang1 results
            else:
                results['-'.join(lang_pair)] = score

    df = pd.DataFrame(
        list(results.items()),
        columns=["language", "spearman cosine"],
    )
    df.to_csv(f'{dataset_name}_modular_results.csv')
