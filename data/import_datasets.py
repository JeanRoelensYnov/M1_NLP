import polars as pl
from pathlib import Path
import shutil
import logging

logger = logging.getLogger()

BOOL_EVAL = ["y", "yes", "oui"]
DATASET_URL = "hf://datasets/adrienheymans/imdb-movie-genres/"
SPLITS = {
    "train": "data/train-00000-of-00001-b7b538a3d562331b.parquet",
    "test": "data/test-00000-of-00001-8b6fe98b0dbe46a8.parquet",
}


def manage_input(case: int) -> bool:
    resp = False
    if case == 1:
        resp = (
            input(
                "Dataset seems to be already saved, would you like to re import it ?\n(y/n): "
            )
            .strip()
            .lower()
            in BOOL_EVAL
        )
        if resp:
            return (
                input(
                    "Folder imdb_dataset and everything in it will be reset are you sure ?\n(y/n): "
                )
                .strip()
                .lower()
                in BOOL_EVAL
            )
        else:
            return
    if case == 2:
        return (
            input(
                "Would you like to create imdb_dataset and data_exploration folder and download corresponding dataset ?\n(y/n): "
            )
            .strip()
            .lower()
            in BOOL_EVAL
        )
    raise ValueError(f"Case {case} not existing.")


def reset_folder(folder_path: Path):
    current_dir = Path(__file__).parent.resolve()
    folder_path = folder_path.resolve()

    if folder_path == current_dir:
        raise ValueError("Refusing to delete the current script directory.")

    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def download_ds(folder_path: Path):
    logger.info("Reading train split from distant repository (HuggingFace)...")
    df_train = pl.read_parquet(DATASET_URL + SPLITS["train"])
    train_path = folder_path / "train.parquet"
    logger.info(f"Saving it to {train_path}")
    df_train.write_parquet(train_path)
    logger.info(
        "train split saved, reading test split from distant repository (HuggingFace)..."
    )
    df_test = pl.read_parquet(DATASET_URL + SPLITS["test"])
    test_path = folder_path / "test.parquet"
    logger.info(f"Saving it to {test_path}")
    df_test.write_parquet(test_path)
    logger.info("test split saved.")


def main():
    # Path to directory
    dir_path = Path(__file__).parent

    # Folders name
    imdb_dataset = "imdb_dataset"
    data_exploration = "data_exploration"

    # Target
    target_folder = dir_path / imdb_dataset
    analysis_folder = dir_path / data_exploration
    # User reponse
    resp = False

    if target_folder.exists() and target_folder.is_dir():
        if manage_input(1):
            reset_folder(target_folder)
            reset_folder(analysis_folder)
            download_ds(target_folder)
        else:
            return
    else:
        if manage_input(2):
            reset_folder(target_folder)
            reset_folder(analysis_folder)
            download_ds(target_folder)
        return


if __name__ == "__main__":
    main()
