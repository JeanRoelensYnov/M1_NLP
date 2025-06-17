"""
File to apply embedding to our train / test datasets, and save the embedding model for later use.
"""

from pathlib import Path
import polars as pl
import logging
from typing import List
from nltk.tokenize import word_tokenize
from generate_tf_idf_matrix import get_clean_text_as_list
from gensim.models import Word2Vec
import nltk

nltk.download("punkt_tab")
# Module variables
PARENT_FOLDER = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_text(df: List[str]) -> List[List[str]]:
    try:
        return [word_tokenize(text) for text in df]
    except Exception as e:
        raise Exception(f"Error while tokenizing : {e}")


def train_word2vec_model(list_token: List[List[str]]) -> Word2Vec:
    model = Word2Vec(
        sentences=list_token, vector_size=100, window=5, min_count=2, workers=4
    )
    return model


def create_embedding_folder():
    embedding_folder = PARENT_FOLDER / "embedding"
    try:
        embedding_folder.mkdir(parents=False, exist_ok=False)
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(
            f"Parents not found while resolving path : {embedding_folder}"
        )
    except FileExistsError as fee:
        logger.info("Directory embedding already existing")
    except Exception as e:
        raise Exception(f"Exception while creating 'embedding' folder : {e}")

def main():
    create_embedding_folder()
    train_df, _ = get_clean_text_as_list()
    tokenized_train = tokenize_text(train_df)

    w2v_model = train_word2vec_model(tokenized_train)

    output_path = PARENT_FOLDER / "embedding" / "w2v.model"

    w2v_model.save(str(output_path))

if __name__ == "__main__":
    main()
