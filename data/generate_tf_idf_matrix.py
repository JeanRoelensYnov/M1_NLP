"""
File to vectorize preprocessed datasets, and then save the TF-IDF matrix
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import polars as pl
import logging
from typing import Tuple, List
import joblib

# Module variables
PARENT_FOLDER = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    try:
        train = pl.read_parquet(
            PARENT_FOLDER / "preprocessed" / "train_preprocessed.parquet"
        )
        test = pl.read_parquet(
            PARENT_FOLDER / "preprocessed" / "test_preprocessed.parquet"
        )
        return train, test
    except Exception as e:
        raise Exception(f"Exception while reading parquet file : {e}")


def get_clean_text_as_list() -> Tuple[List[str], List[str]]:
    try:
        train, test = get_dataframes()
        return train["clean_text"].to_list(), test["clean_text"].to_list()
    except Exception as e:
        raise Exception(f"Exception while retrieving 'clean_text' columns : {e}")


def create_vectorizer_folder():
    vectorizer_path = PARENT_FOLDER / "vectorizer"
    try:
        vectorizer_path.mkdir(parents=False, exist_ok=False)
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(
            f"Parents not found while resolving path : {vectorizer_path}"
        )
    except FileExistsError as fee:
        logger.info("Directory vectorizer already existing")
    except Exception as e:
        raise Exception(f"Exception while creating 'vectorizer' folder : {e}")


if __name__ == "__main__":
    create_vectorizer_folder()
    train_clean, test_clean = get_clean_text_as_list()

    # Compute TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(train_clean)
    X_test_tfidf = tfidf.transform(test_clean)

    # Save the TF-IDF vectorizer (Check README)
    dump_path = PARENT_FOLDER / "vectorizer" / "tfidf_vectorizer.pkl"
    joblib.dump(tfidf, dump_path)
