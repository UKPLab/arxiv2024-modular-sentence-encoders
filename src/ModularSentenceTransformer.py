import json
import logging
import os
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict

from numpy import ndarray
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange

import transformers

from sentence_transformers import __MODEL_HUB_ORGANIZATION__
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import (
    batch_to_device,
    fullname,
)
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers import __version__, SentenceTransformer

from .CustomTransformer import CustomTransformer
from .utils import lang_2_script


logger = logging.getLogger(__name__)

device = torch.device("cuda")

class ModularSentenceTransformer(SentenceTransformer):

    def fit(
        self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        lang_names=None,
        pivot_model=None,
        sonar=False
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        :param lang_names: A list of training languages, e.g. ['eng', deu', ...]
        :param pivot_model: Path of the pivot model. Needed when training cross-lingual adapter. 
        :param sonar: Whether training a SONAR model. 
        """
        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(
                ModelCardTemplate.get_train_objective_info(dataloader, loss)
            )
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {
                "evaluator": fullname(evaluator),
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "scheduler": scheduler,
                "warmup_steps": warmup_steps,
                "optimizer_class": str(optimizer_class),
                "optimizer_params": optimizer_params,
                "weight_decay": weight_decay,
                "evaluation_steps": evaluation_steps,
                "max_grad_norm": max_grad_norm,
            },
            indent=4,
            sort_keys=True,
        )
        self._model_card_text = None
        self._model_card_vars["{TRAINING_SECTION}"] = (
            ModelCardTemplate.__TRAINING_SECTION__.replace(
                "{LOSS_FUNCTIONS}", info_loss_functions
            ).replace("{FIT_PARAMETERS}", info_fit_parameters)
        )

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.to(self.device)

        self.sonar = sonar
        self.lang_names = lang_names

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self.device)

        self.pivot_model_tokenize_func = pivot_model.tokenize
        self.pivot_model = True if pivot_model else False

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = optimizer_class(
                optimizer_grouped_parameters, **optimizer_params
            )
            scheduler_obj = self._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps,
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(
                steps_per_epoch,
                desc="Iteration",
                smoothing=0.05,
                disable=not show_progress_bar,
            ):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    if self.sonar or self.pivot_model:
                        features = list(
                            map(lambda batch: batch_to_device(batch, self.device), features)
                        )    
                    else:                    
                        labels = labels.to(self.device)

                    if use_amp:
                        with torch.autocast(device_type=self.device.type):
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            loss_model.parameters(), max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            loss_model.parameters(), max_grad_norm
                        )
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator,
                        output_path,
                        save_best_model,
                        epoch,
                        training_steps,
                        callback,
                    )

                    for loss_model in loss_models:
                        loss_model.model.zero_grad()
                        loss_model.model.train()

                if (
                    checkpoint_path is not None
                    and checkpoint_save_steps is not None
                    and checkpoint_save_steps > 0
                    and global_step % checkpoint_save_steps == 0
                ):
                    self._save_checkpoint(
                        checkpoint_path, checkpoint_save_total_limit, global_step
                    )

            self._eval_during_training(
                evaluator, output_path, save_best_model, epoch, -1, callback
            )

        if (
            evaluator is None and output_path is not None
        ):  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(
                checkpoint_path, checkpoint_save_total_limit, global_step
            )

    def save(
        self,
        path: str,
        model_name: Optional[str] = None,
        create_model_card: bool = True,
        train_datasets: Optional[List[str]] = None,
        safe_serialization: bool = True,
    ):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders

        :param path: Path on disc
        :param model_name: Optional model name
        :param create_model_card: If True, create a README.md with basic information about this model
        :param train_datasets: Optional list with the names of the datasets used to to train the model
        :param safe_serialization: If true, save the model using safetensors. If false, save the model the traditional PyTorch way
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info("Save model to {}".format(path))

        if self.pivot_model:  # only save target language adapter
            self._first_module().auto_model.save_adapter(path, 'cla') 
            return

        modules_config = []

        # Save some model info
        if "__version__" not in self._model_config:
            self._model_config["__version__"] = {
                "sentence_transformers": __version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            }

        with open(os.path.join(path, "config_sentence_transformers.json"), "w") as fOut:
            config = self._model_config.copy()
            config["prompts"] = self.prompts
            config["default_prompt_name"] = self.default_prompt_name
            json.dump(config, fOut, indent=2)

        # Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if idx == 0 and isinstance(module, CustomTransformer):  # Save transformer model in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            if isinstance(module, CustomTransformer):
                module.save(model_path, safe_serialization=safe_serialization)
            else:
                module.save(model_path)

            modules_config.append(
                {"idx": idx, "name": name, "path": os.path.basename(model_path), "type": type(module).__module__}
            )

        with open(os.path.join(path, "modules.json"), "w") as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def smart_batching_collate(self, batch):
        if self.sonar or self.pivot_model:
            return self.smart_batching_collate_by_lang(batch)
        else:
            return super().smart_batching_collate(batch)

    def smart_batching_collate_by_lang(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...].
        The label for each InputExample is the language of the example.

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            Tokenized sentences dict {lang: tokenized sentences} and indices for the sentences
        """
        texts = [
            example.texts for example in batch
        ]  # [[anchor1, positive1, (optinal: negative1)], ...]
        labels = [
            example.label for example in batch
        ]  # [[anchor1_lang, positive1_lang, (negative1_lang)], ...]

        text_cols = list(zip(*texts))  # [anchors, positivess, (negatives)]
        label_cols = list(
            zip(*labels)
        )  # [anchor1_langs, positive1_langs, (negative1_langs)]

        lang_2_texts = defaultdict(list)  # group texts by langauge. For cross-lingual adapter training, there are 2 languages: eng(pivot) and target language
        lang_2_indices = defaultdict(list)  # store texts' position in batch and col_id
        for col_id in range(
            len(text_cols)
        ):  # 2 (anchor, postive) or 3 (anchor, postive, negative)
            text_batch, label_batch = text_cols[col_id], label_cols[col_id]
            for i in range(len(text_batch)):
                text, lang = text_batch[i], label_batch[i]
                lang_2_texts[lang].append(text)
                lang_2_indices[lang].append([i, col_id])

        # Sort languages so that pivot lang is always the first
        lang_2_texts = {k: lang_2_texts[k] for k in sorted(list(lang_2_texts.keys()))}
        lang_2_indices = {k: lang_2_indices[k] for k in sorted(list(lang_2_indices.keys()))}

        sentence_features = []  # tokenized texts for each language
        for lang, texts_to_tokenize in lang_2_texts.items():
            if self.sonar:
                self.tokenizer.src_lang = lang_2_script[self.lang_names[lang]]
                sentence_features.append(self.tokenize(texts_to_tokenize))
            else:  # for bilingual alignment
                if lang == 0:  # pivot lang
                    sentence_features.append(self.pivot_model_tokenize_func(texts_to_tokenize))
                else:
                    sentence_features.append(self.tokenize(texts_to_tokenize))

        labels = list(lang_2_indices.values())
        return sentence_features, labels

    def encode(
        self,
        sentences: Union[str, List[str]],
        lang: str = None,
        **kwargs,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings.

        :param sentences: the sentences to embed.
        :param lang: language of the sentences, needed when the model's tokenizer needs a src_lang param.  
        :param prompt_name: The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
            which is either set in the constructor or loaded from the model configuration. For example if
            `prompt_name` is ``"query"`` and the `prompts` is ``{"query": "query: ", ...}``, then the sentence "What
            is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
            is appended to the prompt. If `prompt` is also set, this argument is ignored.
        :param prompt: The prompt to use for encoding. For example, if the prompt is ``"query: "``, then the
            sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
            because the sentence is appended to the prompt. If `prompt` is set, `prompt_name` is ignored.
        :param batch_size: the batch size used for the computation.
        :param show_progress_bar: Whether to output a progress bar when encode sentences.
        :param output_value: The type of embeddings to return: "sentence_embedding" to get sentence embeddings,
            "token_embeddings" to get wordpiece token embeddings, and `None`, to get all output values. Defaults
            to "sentence_embedding".
        :param precision: The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or
            "ubinary". All non-float32 precisions are quantized embeddings. Quantized embeddings are smaller in
            size and faster to compute, but may have a lower accuracy. They are useful for reducing the size
            of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
        :param convert_to_numpy: Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
        :param convert_to_tensor: Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
        :param device: Which `torch.device` to use for the computation.
        :param normalize_embeddings: Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned. If only one string
            input is provided, then the output is a 1d array with shape [output_dimension]. If `convert_to_tensor`, a
            torch Tensor is returned instead. If `self.truncate_dim <= output_dimension` then output_dimension is
            `self.truncate_dim`.
        """
        if lang:
            self.tokenizer.src_lang = lang
        return super().encode(sentences, **kwargs)
