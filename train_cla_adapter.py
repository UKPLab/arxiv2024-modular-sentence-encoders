import logging
import os
import argparse

import adapters
from adapters import ParBnConfig

from sentence_transformers import LoggingHandler, models

from src.MultiDatasetDataloader import MultiDatasetDataLoader
from src.ParaphraseDataset import ParaphraseDataset
from src.CustomMNRL import CrossModelMNRL
from src.ModularSentenceTransformer import ModularSentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--target_model_name_or_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--pivot_model_name_or_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
)

parser.add_argument(
    "--output_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--max_seq_length",
    type=int,
    default=128,
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=1,
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
)

parser.add_argument(
    "--train_batch_size",
    type=int,
    default=128,
)

parser.add_argument(
    "--target_lang",
    type=str,
    default=None,
)

parser.add_argument(
    "--pivot_lang",
    type=str,
    default="eng",
)

args = parser.parse_args()

pivot_lang = args.pivot_lang
target_lang = args.target_lang
target_model_name_or_path = args.target_model_name_or_path
pivot_model_name_or_path = args.pivot_model_name_or_path
max_seq_length = args.max_seq_length
lr = args.learning_rate
train_batch_size = args.train_batch_size
num_epochs = args.num_epochs
output_path = args.output_path
data_dir = args.train_data_dir


# Create pivot model
pivot_model = ModularSentenceTransformer(model_name_or_path=pivot_model_name_or_path)
for param in pivot_model.parameters():
    param.requires_grad = False
pivot_model.max_seq_length = max_seq_length

# Create target model
word_embedding_model = models.Transformer(
    target_model_name_or_path, max_seq_length=max_seq_length
)

# Add adapter to the target language model
adapters.init(word_embedding_model.auto_model)
adapter_config = ParBnConfig()
word_embedding_model.auto_model.add_adapter("cla", adapter_config)
word_embedding_model.auto_model.set_active_adapters("cla")
word_embedding_model.auto_model.train_adapter("cla")

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
target_model = ModularSentenceTransformer(modules=[word_embedding_model, pooling_model])


###### Load Paraphrase Datasets ######
train_datasets = []
for dataset_dir in os.listdir(data_dir):
    dataset_path = f"{data_dir}/{dataset_dir}"
    dataset = ParaphraseDataset(
        langs=[pivot_lang, target_lang], dir_path=dataset_path
    )  # pivot lang should always be the first
    train_datasets.append(dataset.data)

train_dataloader = MultiDatasetDataLoader(
    train_datasets,
    batch_size=train_batch_size,
    num_langs=2,
    batch_type="cross",
)

###### Train Cross-Lingual Adapter on the Target Language Model ######

train_loss = CrossModelMNRL(
    model=target_model, pivot_model=pivot_model, batch_size=train_batch_size
)

target_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    scheduler="constantlr",
    output_path=output_path,
    optimizer_params={"lr": lr, "eps": 1e-6},
    pivot_model=pivot_model,
)
####
