from pathlib import Path
import os
import polars as pl
import pyarrow as pa
from data.datasets_slicing import retrieve_full_path
from datasets import Dataset

PARENT_FOLDER = Path(__file__).parent


if __name__ == "__main__":

    train_arrow_filepath = retrieve_full_path("train")
    test_arrow_filepath = retrieve_full_path("test")

    # Once we have both files we want to save them as CSV for our data exploration in jupyter notebook
    # this will allow us to see what transformation exactly we want to perform on our data
    train_ds = Dataset.from_file(str(train_arrow_filepath)).to_polars()
    test_ds = Dataset.from_file(str(test_arrow_filepath)).to_polars()

    train_ds_for_analysis = train_ds.slice(0, 10_000)
    test_ds_for_analysis = test_ds.slice(0, 10_000)

    train_ds_for_analysis.write_csv(
        PARENT_FOLDER / "data_exploration" / "train.csv", separator=","
    )
    test_ds_for_analysis.write_csv(
        PARENT_FOLDER / "data_exploration" / "test.csv", separator=","
    )
