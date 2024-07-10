"""
This script uses FOCUS to train a language-specific tokenizer and initialize a new embedding matrix based on the multilingual embedding matrix of the source model.  

FOCUS official code: https://github.com/konstantinjdobler/focus
FOCUS paper: https://aclanthology.org/2023.emnlp-main.829/
"""

import argparse

import torch

from deepfocus import FOCUS

from transformers import AutoModelForMaskedLM, AutoTokenizer


# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_name",
    type=str,
    default=None,
)

parser.add_argument(
    "--train_data_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--tokenizer_save_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--embedding_save_path",
    type=str,
    default=None,
)

args = parser.parse_args()

model_name = args.model_name
train_data_path = args.train_data_path
tokenizer_save_path = args.tokenizer_save_path
embedding_save_path = args.embedding_save_path

# Load source model and its tokenizer
source_tokenizer = AutoTokenizer.from_pretrained(model_name)
source_model = AutoModelForMaskedLM.from_pretrained(model_name)

# Load training data
with open(train_data_path, encoding="utf-8") as f:
    train_data = [line.strip() for line in f]

# Train and save new tokenizer
target_tokenizer = source_tokenizer.train_new_from_iterator(
    train_data, vocab_size=50_048
)
target_tokenizer.save_pretrained(tokenizer_save_path)

## Initialize and save new embeddings
target_embeddings = FOCUS(
    fasttext_model_dim=300,
    source_embeddings=source_model.get_input_embeddings().weight,
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    target_training_data_path=train_data_path,
)
torch.save(target_embeddings, embedding_save_path)
