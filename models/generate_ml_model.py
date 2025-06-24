"""
File to generate the ML model which I'll use to predict the genre of a synopsis
"""

from pathlib import Path
import logging
import joblib
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# Since the dataset is already split between train and test there's no use (at least I think) to import train_test_split

# logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
# Since we want the best result I'm going to drop the pre-made tfidf vectorizer
# VECTORIZER_PATH = PROJECT_ROOT / "data" / "vectorizer" / "tfidf_vectorizer.pkl"
TRAIN_PARQUET = PROJECT_ROOT / "data" / "preprocessed" / "train_preprocessed.parquet"
TEST_PARQUET = PROJECT_ROOT / "data" / "preprocessed" / "test_preprocessed.parquet"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "ml_model.pkl"


def load_data():
    # Retrieve train and test df
    logger.info("Loading datasets...")
    train_df = pl.read_parquet(TRAIN_PARQUET)
    test_df = pl.read_parquet(TEST_PARQUET)

    X_train = train_df["clean_text"].to_list()
    y_train = train_df["genre"].to_list()

    X_test = test_df["clean_text"].to_list()
    y_test = test_df["genre"].to_list()

    return X_train, X_test, y_train, y_test


def build_pipeline():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, solver="saga")),
        ]
    )


def get_param_grid():
    return {
        "tfidf__max_features": [10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__class_weight": [None, "balanced"],
    }


def train_and_evaluate():
    # Load data and vectorizer
    X_train, X_test, y_train, y_test = load_data()
    pipeline = build_pipeline()
    param_grid = get_param_grid()

    logger.info("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=2
    )

    grid_search.fit(X_train, y_train)

    logger.info("Best parameters found:")
    logger.info(grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info("Classification report:")
    logger.info(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(best_model, MODEL_SAVE_PATH)
    logger.info(f"Model saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
