from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import os

from src.utils import lang_2_script

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Other datasets used in the paper includes: sentence-transformers/simple-wiki, sentence-transformers/altlex, sentence-transformers/quora-duplicates, facebook/xnli
dataset_name = 'sentence-transformers/sentence-compression' # an example
dataset = load_dataset(dataset_name, split='train')
text1 = dataset['text']  # Change it to corresponding column according to the specific dataset
text2 = dataset['simplified']  # Change it to corresponding column according to the specific dataset
output_dir = 'data/paraphrase/sentence-compression'
if not os.path.isdir(output_dir):
    os.makedir(output_dir)

# Load MT model
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-3.3B", use_auth_token=True, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-3.3B", use_auth_token=True).to(device)


def translate_texts(texts, tgt_lang, out_path, batch_size=32):
    dataset = Dataset.from_dict({'text': texts})
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with open(out_path, 'w', encoding='utf-8') as f:
        for batch in dataloader:
            inputs = tokenizer(
                batch['text'], return_tensors="pt", padding=True).to(device)

            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_2_script[tgt_lang]], max_length=130
            )
            text = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True)
            for sent in text:
                f.write(sent + '\n')

# Translate data
langs = ['deu']

for target_lang in langs:
    translate_texts(text1, target_lang,
                    f'{output_dir}/{dataset}/{target_lang}_1.txt', 
                    batch_size=32)
    translate_texts(text2, target_lang,
                    f'{output_dir}/{dataset}/{target_lang}_2.txt', 
                    batch_size=32)
