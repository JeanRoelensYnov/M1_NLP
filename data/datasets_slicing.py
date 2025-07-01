from pathlib import Path
import polars as pl

PARENT_FOLDER = Path(__file__).parent

if __name__ == "__main__":

    imdb_folderpath = PARENT_FOLDER / "imdb_dataset"
    train_df = pl.read_parquet(imdb_folderpath / "train.parquet")
    test_df = pl.read_parquet(imdb_folderpath / "test.parquet")

    # Slice
    train_df = train_df.slice(0, 10_000)
    test_df = test_df.slice(0, 10_000)

    # And save in the correct folder and as csv
    target_folderpath = PARENT_FOLDER / "data_exploration"

    train_df.write_csv(target_folderpath / "train.csv")
    test_df.write_csv(target_folderpath / "test.csv")
