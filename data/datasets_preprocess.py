"""
File to retrieve test.parquet and train.parquet to preprocess them in a similar way to the exploration done in data_exploration.ipnyb.
The fact that both file are parquet make me hope that the process will be relatively fast.
"""

from pathlib import Path
import polars as pl
import logging
from typing import Tuple, List
from num2words import num2words
import re
import nltk

nltk.download("stopwords")
from string import punctuation
import spacy
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PARENT_FOLDER = Path(__file__).parent
CHUNK_SIZE = 5000
# Here I want the model with the more accuracy eventhough it will take time
NLP = spacy.load("en_core_web_trf")


def create_preprocess_folder():
    preprocess_path = PARENT_FOLDER / "preprocessed"
    try:
        preprocess_path.mkdir(parents=False, exist_ok=False)
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(
            f"Parents not found while resolving path : {preprocess_path}"
        )
    except FileExistsError as fee:
        logger.info("Directory preprocessed already existing")
    except Exception as e:
        raise Exception(f"Exception while creating 'preprocessed' folder : {e}")


def get_parquet_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    try:
        train = pl.read_parquet(PARENT_FOLDER / "imdb_dataset" / "train.parquet")
        test = pl.read_parquet(PARENT_FOLDER / "imdb_dataset" / "test.parquet")
        return train, test
    except Exception as e:
        raise Exception(f"Exception while reading parquet file : {e}")


def convert_ordinals(text: str) -> str:
    def ordinal_replacer(match: re.Match):
        number = int(match.group(1))
        return num2words(number, to="ordinal")

    return re.sub(
        r"\b(\d+)(st|nd|rd|th)\b", ordinal_replacer, text
    )  # Regex : https://regex101.com/r/iyqb2p/1


def convert_cardinals(text: str) -> str:
    def cardinal_replacer(match: re.Match):
        number = int(match.group(1))
        return num2words(number)

    return re.sub(r"(\d+)", cardinal_replacer, text)  # Regex capture all number


def convert_all_numbers(text: str) -> str:
    # first convert all ordinals numbers
    text = convert_ordinals(text)
    # And then cardinals
    return convert_cardinals(text)


def save_parquet_dataframe(df: pl.DataFrame, name: str):
    preprocess_path = PARENT_FOLDER / "preprocessed" / f"{name}_preprocessed.parquet"
    try:
        df.write_parquet(preprocess_path)
    except Exception as e:
        raise Exception(f"Exception while writing parquet file : {e}")


def process_chunk(df: pl.DataFrame) -> pl.DataFrame:
    texts = df["text"].to_list()

    cleaned_texts = [convert_all_numbers(t) for t in texts]
    cleaned_texts = [
        re.sub(rf"[{re.escape(punctuation)}]", "", t) for t in cleaned_texts
    ]
    cleaned_texts = [re.sub(r"\s+", " ", t) for t in cleaned_texts]

    processed_texts = []
    for doc in NLP.pipe(cleaned_texts, batch_size=32):
        processed_texts.append(
            " ".join(token.lemma_ for token in doc if not token.is_stop)
        )

    return df.with_columns(pl.Series("clean_text", processed_texts))


if __name__ == "__main__":
    create_preprocess_folder()
    train_df, test_df = get_parquet_dataframes()
    dict_df = {"train": train_df, "test": test_df}
    logger.info("train and test parquet file loaded.")
    start_time = time.time()

    for name, df in dict_df.items():
        chunk_list: List[pl.DataFrame] = []
        chunk_count = 0
        total_chunk = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(0, len(df), CHUNK_SIZE):
            df_chunk = df.slice(i, CHUNK_SIZE)
            chunk_count += 1
            logger.info(
                f"Dataframe: {name} processing Chunk: {chunk_count}/{total_chunk}"
            )
            df_chunk = process_chunk(df_chunk)
            chunk_list.append(df_chunk)
            logger.info(f"Chunk processed.")
        whole_df = pl.concat(chunk_list)
        save_parquet_dataframe(whole_df, name)
        logger.info(
            f"Dataframe: {name} processed in: {time.time() - start_time:.2f} seconds"
        )
