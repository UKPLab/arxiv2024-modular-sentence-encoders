import os
import logging
import argparse

from sentence_transformers import LoggingHandler, models

from src.MultiDatasetDataloader import MultiDatasetDataLoader
from src.ParaphraseDataset import ParaphraseDataset
from src.CustomMNRL import CustomMNRL
from src.ModularSentenceTransformer import ModularSentenceTransformer
from src.CustomTransformer import CustomTransformer

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
    "--model_name_or_path",
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
    default=2e-5,
)

parser.add_argument(
    "--train_batch_size",
    type=int,
    default=128,
)

parser.add_argument(
    "--langs",
    nargs="+",
    help="<Required> Languages of training datasets",
    required=True,
)

parser.add_argument(
    "--train_type",
    type=str,
    default=None,
)


parser.add_argument("--sonar", action="store_true", help="The model is a SONAR model or not.")

args = parser.parse_args()

langs = args.langs
model_name_or_path = args.model_name_or_path
max_seq_length = args.max_seq_length
lr = args.learning_rate
train_batch_size = args.train_batch_size
num_epochs = args.num_epochs
output_path = args.output_path
data_dir = args.train_data_dir
train_type = args.train_type
sonar = args.sonar

######## Create Sentence Transformer  ########
logger.info("Create sentence transformer")

word_embedding_model = CustomTransformer(
    model_name_or_path,
    max_seq_length=max_seq_length,
)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = ModularSentenceTransformer(modules=[word_embedding_model, pooling_model])

###### Load Paraphrase Datasets ######
logger.info("Load paraphrase datasets")
train_datasets = []
for dataset_dir in os.listdir(data_dir):
    dataset_path = f"{data_dir}/{dataset_dir}"
    if os.path.isdir(dataset_path):
        dataset = ParaphraseDataset(langs=langs, dir_path=dataset_path)
        train_datasets.append(dataset.data)

train_dataloader = MultiDatasetDataLoader(
    train_datasets,
    batch_size=train_batch_size,
    num_langs=len(langs),
    batch_type=train_type,
)

###### Train the model ######
train_loss = CustomMNRL(model=model, batch_size=train_batch_size, sonar=sonar)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    scheduler="constantlr",
    output_path=output_path,
    optimizer_params={"lr": lr, "eps": 1e-6},
    lang_names=langs,
    sonar=sonar,
)
