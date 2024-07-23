# Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment
[![Arxiv](https://img.shields.io/badge/Arxiv-2407.14878-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.14878)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)


This repository implements the method for training cross-lingually aligned language-specific sentence encoders in the paper *Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment*.

> **Abstract:** Multilingual sentence encoders are commonly obtained by training multilingual language models to map sentences from different languages into a shared semantic space. As such, they are subject to curse of multilinguality, a loss of monolingual representational accuracy due to parameter sharing. Another limitation of multilingual sentence encoders is the trade-off between monolingual and cross-lingual performance. Training for cross-lingual alignment of sentence embeddings distorts the optimal monolingual structure of semantic spaces of individual languages, harming the utility of sentence embeddings in monolingual tasks. In this work, we address both issues by modular training of sentence encoders, i.e., by separating monolingual specialization from cross-lingual alignment. We first efficiently train language-specific sentence encoders to avoid negative interference between languages (i.e., the curse). We then align all non-English monolingual encoders to the English encoder by training a cross-lingual alignment adapter on top of each, preventing interference with monolingual specialization from the first step. In both steps, we resort to contrastive learning on machine-translated paraphrase data. Monolingual and cross-lingual evaluations on semantic text similarity/relatedness and multiple-choice QA render our modular solution more effective than multilingual sentence encoders, especially benefiting low-resource languages.

![modular_sentence_encoders](https://github.com/UKPLab/arxiv2024-modular-sentence-encoders/assets/56653470/29cdaed4-fe8c-4d34-b04f-dacbd2f2dd94)


Contact person: [Yongxin Huang](mailto:yongxin.huang@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Getting Started
This project requires Python 3.10. To install the requirements: 
  ```bash
  pip install -r requirements.txt
  ```

## Training modular sentence encoders
### Monolingual specialization

#### Step 1: Train Language-specific Tokenizer and Embeddings
The script [`focus.py`](focus.py) is used for training language-specific tokenizers and initializing the new embeddings with FOCUS:

```bash
python focus.py \
    --model_name sentence-transformers/LaBSE \
    --train_data_path data/language_adaptation/deu_train.txt \
    --tokenizer_save_path model/tokenizer/deu_labse \
    --embedding_save_path model/embedding/deu_labse \
```
Parameter description
- `--model_name` is the model name or path of the multilingual source model.
- `--train_data_path` is the path of the .txt file with monolingual text data.
- `--tokenizer_save_path` is the path to save the new tokenizer.
- `--embedding_save_path` is the path to save the new embedding matrix.

#### Step 2: Language Adaptation (LA)
The script [`train_mlm.py`](train_mlm.py) is used for continual monolingual MLM pre-training on each monolingual model:

```bash
python train_mlm.py \
    --train_file data/language_adaptation/deu_train.txt \
    --validation_file data/language_adaptation/deu_val.txt \
    --model_name_or_path sentence-transformers/LaBSE \
    --tokenizer_name model/tokenizer/deu_labse \
    --embedding_path model/embedding/deu_labse  \
    --max_seq_length 256 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --output_dir model/mlm_model/deu \
```
Parameter description
- `--train_file` is the path of the .txt file with monolingual text data for training.
- `--val_file` is the path of the .txt file with monolingual text data for validation.
- `--model_name_or_path` is the model name or path of the multilingual source model.
- `--tokenizer_name` is the path to monolingual tokenizer created in Step 1.
- `--embedding_path` is the path to monolingual embedding matrix created in Step 1.
- `--output_dir` is the path to save the adapted model. 

#### Step 3: Sentence Encoding Training
First, we need to create monolingual paraphrase data in each target language with machine translation. The script [data/paraphrase/translate_paraphrase_data.py](data/paraphrase/translate_paraphrase_data.py) provides code for the translation.   


After we have prepared the training data, we can use the script [train_sentence_encoder.py](train_sentence_encoder.py) to do monolingual sentence embedding training:

```bash
python train_sentence_encoder.py \
    --langs deu \
    --model_name_or_path model/mlm_model/deu \
    --max_seq_length 64 \
    --learning_rate 2e-5 \
    --train_batch_size 32 \
    --num_epochs 1 \
    --output_path model/mono_encoder/labse_deu \
    --train_data_dir data/paraphrase \
    --train_type cross \
```
Parameter description
- `--langs` is the languages of training data. For monolingual specialization, we always use one target language. For training multilingual baselines, you can pass multiple languages, e.g. `--langs eng deu`.
- `--model_name_or_path` is the model name or path of the model trained in Step 2.  
- `--output_path` is the path to save the monolingual sentence encoder. 
- `--train_data_dir` is the directory contraining the paraphrase data files created by the script [data/paraphrase/translate_paraphrase_data.py](data/paraphrase/translate_paraphrase_data.py).
- `--train_type` should be either "mono" (monolingual) or "cross" (cross-lingual). For monolingual specialization, we always set it to "mono". "cross" can be used for training multilingual baselines, such as Single<sub>c</sub>. 

### Cross-Lingual Alignment
After training monolingual sentence encoders, we train cross-lingual adapters to align them using the script [train_cla_adapter.py](train_cla_adapter.py):  

```bash
python train_cla_adapter.py \
    --pivot_lang eng \
    --target_lang deu \
    --pivot_model_name_or_path model/mono_encoder/labse_eng \
    --target_model_name_or_path model/mono_encoder/labse_deu \
    --max_seq_length 128 \
    --learning_rate 1e-4 \
    --train_batch_size 128 \
    --num_epochs 1 \
    --output_path model/cla_adapter/labse_deu_adapter \
    --train_data_dir data/paraphrase \
```
Parameter description
- `--pivot_lang`: We always use English as our pivot language and align each non-English encoder to the English encoder. 
- `--pivot_model_name_or_path` is the name or path of the pivot language model, i.e. the English encoder trained in Step 3 in monolingual specialization. 
- `--target_lang` is the language of the encoder that should be aligned to the English encoder.
- `--target_model_name_or_path` is the name or path to the target language model trained in Step 3 in monolingual specialization.
- `--output_path` is the path to save the cross-lingual alignment adapter (CLA adapter). 
- `--train_data_dir` is the directory contraining the paraphrase data files created by the script [data/paraphrase/translate_paraphrase_data.py](data/paraphrase/translate_paraphrase_data.py).

## Evaluation
### Semantic Textual Similarity/Relatedness
The script [eval_sts_str.py](eval_sts_str.py) is used for the evaluation on the STS and STR datasets. See [this README](data/sts_str/README.md) and the script [eval_data_utils.py](eval_data_utils.py) on how to download and use the evaluation data.

### Multiple-Choice Question Answering
The script [eval_belebele.py](eval.belebele.py) is for the evaluation on Belebele.  

## Cite

Please use the following citation:

```
@article{huang2024modularsentenceencodersseparating,
      title={Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment}, 
      author={Yongxin Huang and Kexin Wang and Goran Glavaš and Iryna Gurevych},
      year={2024},
      url={https://arxiv.org/abs/2407.14878},
      journal={ArXiv preprint},
      volume={abs/2407.14878},
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
