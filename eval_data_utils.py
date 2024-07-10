import io
import zipfile

import pandas as pd

from datasets import load_dataset

from umsc import UgMultiScriptConverter


lang2code_to_lang3code = {
    "en": "eng",
    "cs": "ces",
    "fr": "fra",
    "de": "deu",
    "tr": "tur",
    "es": "spa",
    "ar": "arb",
    "it": "ita",
    "nl": "nld",
    "ko": "kor",
    "pl": "pol",
    "ru": "rus",
    "zh": "zho",
    "az": "azj",
    "kk": "kaz",
    "ky": "kir",
    "ug": "uig",
    "uz": "uzn",
}

STR_LANGS = ["eng", "tel", "hau", "amh", "mar", "kin"]
STS17_LANGS = [
    "ar-ar",
    "en-ar",
    "en-de",
    "en-en",
    "en-tr",
    "es-en",
    "es-es",
    "fr-en",
    "it-en",
    "ko-ko",
    "nl-en",
]
STS17_ADDITIONAL_LANGS = [
    "de-de",
    "fr-fr",
    "cs-cs",
    "de-en",
    "en-fr",
    "en-cs",
    "cs-en",
    "de-fr",
    "fr-de",
    "cs-de",
    "de-cs",
    "cs-fr",
    "fr-cs",
]
STS22_LANGS = [
    "ar",
    "de",
    "de-en",
    "de-fr",
    "de-pl",
    "en",
    "es",
    "es-en",
    "es-it",
    "fr",
    "fr-pl",
    "it",
    "pl",
    "pl-en",
    "ru",
    "tr",
    "zh",
    "zh-en",
]


def load_str24_data(path="data/sts_str/Semantic_Relatedness_SemEval2024"):
    """
    Datasets should have been downloaded into the path by
    git clone https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024.git
    """
    all_data = dict()
    for lang in STR_LANGS:
        data = {"sentences1": [], "sentences2": [], "scores": []}
        df_str_rel = pd.read_csv(
            f"{path}/Track A/{lang}/{lang}_test_with_labels.csv", encoding="utf-8"
        )
        for text, score in zip(df_str_rel["Text"], df_str_rel["Score"]):
            try:
                sent1, sent2 = text.split("\n")
            except:
                sent1, sent2 = text.split("\t")
            data["sentences1"].append(sent1)
            data["sentences2"].append(sent2)
            data["scores"].append(score)
            all_data[(lang, lang)] = data
    return all_data


def load_sts17_data():
    """
    Load STS17 data from HuggingFace.
    """
    all_data = dict()
    for lang_pair in STS17_LANGS:
        dataset = load_dataset("mteb/sts17-crosslingual-sts", lang_pair)["test"]
        lang1, lang2 = lang_pair.split("-")
        data = {
            "sentences1": dataset["sentence1"],
            "sentences2": dataset["sentence2"],
            "scores": dataset["score"],
        }
        all_data[(lang2code_to_lang3code[lang1], lang2code_to_lang3code[lang2])] = data

    return all_data


def load_additional_sts17_data(path="data/sts_str/cross-lingual-sts"):
    """
    Datasets should have been downloaded into the path by
    git clone https://gitlab.com/tigi.cz/cross-lingual-sts.git
    """

    def read_zipped_file(filename):
        fIn = zip.open(filename)
        return [line.strip() for line in io.TextIOWrapper(fIn, "utf8")]

    # Read monolingual data
    with zipfile.ZipFile(f"{path}/dataset.zip") as zip:
        scores = read_zipped_file(f"dataset/STS.2017.gs.track5.first-second.txt")
        scores = [float(s) for s in scores]

        lang_2_sents = dict()
        for lang in ["EN", "CS", "DE", "FR"]:
            sents1 = read_zipped_file(f"dataset/STS.2017.input.track5.{lang}.first.txt")
            sents2 = read_zipped_file(
                f"dataset/STS.2017.input.track5.{lang}.second.txt"
            )
            lang_2_sents[lang.lower()] = (sents1, sents2)

    # Create data for language pairs
    all_data = dict()
    for lang_pair in STS17_ADDITIONAL_LANGS:
        lang1, lang2 = lang_pair.split("-")
        data = {
            "sentences1": lang_2_sents[lang1][0],
            "sentences2": lang_2_sents[lang2][1],
            "scores": scores,
        }
        all_data[lang2code_to_lang3code[lang1], lang2code_to_lang3code[lang2]] = data
    return all_data


def load_kardes_data(path="data/sts_str/Kardes-NLU"):
    """
    Datasets should have been downloaded into the path by
    git clone https://github.com/lksenel/Kardes-NLU.git
    """
    lang_2_code = {
        "azeri": "az",
        "kazakh": "kk",
        "kyrgyz": "ky",
        "uyghur": "ug",
        "uzbek": "uz",
    }

    lang_2_data = dict()
    df = pd.read_csv(f"{path}/Data/azeri/sts.test.az.csv")
    lang_2_data["eng"] = (df["sentence1"], df["sentence2"])
    scores = df["score"]

    # Uyghur transliterator
    source_script = "UCS"
    target_script = "UAS"
    converter = UgMultiScriptConverter(source_script, target_script)

    for lang, code in lang_2_code.items():
        df = pd.read_csv(f"{path}/Data/{lang}/sts.test.{code}.csv")
        lang_2_data[lang2code_to_lang3code[code]] = (
            (
                [converter(text) for text in df["s1_translation"]]
                if code == "ug"
                else df["s1_translation"]
            ),
            (
                [converter(text) for text in df["s2_translation"]]
                if code == "ug"
                else df["s2_translation"]
            ),
        )

    all_data = dict()
    for lang1 in lang_2_data.keys():
        for lang2 in lang_2_data.keys():
            data = {
                "sentences1": lang_2_data[lang1][0],
                "sentences2": lang_2_data[lang2][1],
                "scores": scores,
            }
            all_data[(lang1, lang2)] = data
    return all_data


def load_sts22_data():
    """
    Load STS22 data from HuggingFace
    """
    all_data = dict()
    for lang in STS22_LANGS:
        dataset = load_dataset("mteb/sts22-crosslingual-sts", lang)["test"]

        if "-" in lang:
            lang1, lang2 = lang.split("-")
        else:
            lang1 = lang2 = lang
        data = {
            "sentences1": dataset["sentence1"],
            "sentences2": dataset["sentence2"],
            "scores": dataset["score"],
        }
        all_data[(lang2code_to_lang3code[lang1], lang2code_to_lang3code[lang2])] = data

    return all_data


def load_belebele_data(langs):
    dataset = load_dataset("facebook/belebele")

    cols = [
        "flores_passage",
        "question",
        "mc_answer1",
        "mc_answer2",
        "mc_answer3",
        "mc_answer4",
        "correct_answer_num",
    ]

    # Create parallel datasets
    dfs = [
        dataset[lang]
        .to_pandas()
        .set_index(["link", "question_number"])[cols]
        .rename(columns={k: f"{k}_{lang}" for k in cols})
        for lang in langs
    ]
    df = dfs[0].join(dfs[1:])

    df = df.rename(columns={f"correct_answer_num_{langs[0]}": "correct_answer_num"})

    return df[
        [
            col
            for col in df.columns
            if col not in [f"correct_answer_num_{lang}" for lang in langs]
        ]
    ]
