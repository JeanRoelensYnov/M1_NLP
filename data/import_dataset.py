from datasets import load_dataset
from pathlib import Path
import shutil

from torch import Value

BOOL_EVAL = ["y", "yes", "oui"]
DATASET_NAME = "adrienheymans/imdb-movie-genres"


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
    ds = load_dataset(DATASET_NAME)
    ds.save_to_disk(str(folder_path))


def main():
    # Path to directory
    dir_path = Path(__file__).parent

    # Folder name
    folder_name = "imdb_dataset"

    # Target
    target_folder = dir_path / folder_name

    # User reponse
    resp = False

    if target_folder.exists() and target_folder.is_dir():
        if manage_input(1):
            reset_folder(target_folder)
            download_ds(target_folder)
        else:
            return
    else:

        pass


if __name__ == "__main__":
    main()
