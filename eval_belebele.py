from collections import defaultdict

from sentence_transformers import util

import torch

import pandas as pd

from src.utils import lang_2_script

from eval_utils import load_adapter_model
from eval_data_utils import load_belebele_data


device = torch.device('cuda')


langs = [  
    "eng",
    "ces",
    "fra",
    "deu",
    "azj",  
    "kaz",
    "kir",
    "uzn",
    "spa",
    "tel",
    "hau",
    "amh",
    "mar",
    "ita",
    "nld",
    "kor",
    "kin",
    'arb',
    'tur',
    "pol", 
    "rus",
    "zho",
]


def load_lang_model(lang):
    """
    Load the monolingual sentence encoder. 
    Assume monolingual models are stored with the same name format. 
    """
    return load_adapter_model(
        model_path=f'model/mono_encoder/labse_{lang}',
        adapter_path=f'model/cla_adapter/labse_{lang}_adapter' if lang!='eng' else None
    )


# Load evaluation data
dataset = load_belebele_data([lang_2_script[lang] for lang in langs])

# Encode data with monolingual models
qa_pair_embs = defaultdict(dict)
passage_embs = defaultdict(dict)
for lang in langs:
    model = load_lang_model(lang).to(device)

    questions = dataset[f'question_{lang_2_script[lang]}']
    options = [dataset[f'mc_answer{i}_{lang_2_script[lang]}'] for i in range(1, 5)]

    passage = dataset[f'flores_passage_{lang_2_script[lang]}']   

    qa_pairs = []
    for i in range(len(questions)):
        question = questions.iloc[i]
        for option_list in options:  
            qa_pairs.append(f'{question} {option_list.iloc[i]}')

    # Encode data without cross-lingual adapter
    passage_embs[lang]['mono_emb'] = model.encode(passage, batch_size=512)
    qa_pair_embs[lang]['mono_emb'] = model.encode(qa_pairs, batch_size=512)

    # Encode data with cross-lingual adapter
    if lang != 'eng':
        model[0].auto_model.set_active_adapters('cla')
    passage_embs[lang]['cross_emb'] = model.encode(passage, batch_size=512)
    qa_pair_embs[lang]['cross_emb'] = model.encode(qa_pairs, batch_size=512)   


# Evaluate models
answers = torch.tensor([int(i)-1 for i in dataset['correct_answer_num']]).to(device)
dim = passage_embs[langs[0]]['mono_emb'].shape[1]

results = dict()
for lang1 in langs:
    for lang2 in langs:
        emb_type = 'mono_emb' if lang1 == lang2 else 'cross_emb'

        # Get embeddings of QA pairs in lang1
        qa_pair_emb = qa_pair_embs[lang1][emb_type].reshape(-1, 4, dim).transpose(1, 0, 2)

        # Get embeddings of passages in lang2
        passage_emb = passage_embs[lang2][emb_type].reshape(1, -1, dim)

        # Similarity of passage and each QA pair
        scores = util.pairwise_cos_sim(qa_pair_emb, passage_emb)
        
        preds = scores.argmax(dim=0)

        correct = (preds.to(device)==answers.to(device)).sum()
        acc = round(correct.item() / preds.shape[0] * 100, 2)

        lang_pair = '-'.join([lang1, lang2])
        rev_pair = '-'.join([lang2, lang1])
        if rev_pair not in results:
            results[lang_pair] = acc
        else:
            results[rev_pair] = round((results[rev_pair] + acc) / 2, 2)

# Save results
df = pd.DataFrame(list(results.items()), columns=['lang', 'acc'])
df.to_csv('belebele_results.csv')
