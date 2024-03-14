"""Convert the BUS-BRA dataset into ImageFolder format.

Note: To correctly stratify the dataset the folds are generated at this step (as
opposed to in the training script) so that each patient is in only one fold.

Usage:
    python bus_bra.py
        --input_dir /path/to/dataset
        --output_dir /path/to/output
        [--birads]
"""

import argparse
import os
import shutil

import pandas as pd

CLINICAL_DATA_FILENAME = "bus_data.csv"


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(prog="bus_bra.py")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="The path to the original dataset directory",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to write the newly formatted dataset to",
        required=True,
    )
    parser.add_argument(
        "--birads",
        action="store_true",
        help="Use BIRADS classification as the target variable",
    )

    args = parser.parse_args()
    assert os.path.isdir(args.input_dir), "input_dir must be an existing directory"
    assert not os.path.exists(args.output_dir), "output_dir must not already exist"

    return args


def main():
    """Convert the BUS-BRA dataset into ImageFolder format."""
    args = parse_args()

    # Read CSV file with clinical data
    df = pd.read_csv(os.path.join(args.input_dir, CLINICAL_DATA_FILENAME))
    df["Patient"] = df["ID"].str.split("-").str[0]
    df["Pathology"] = df["Pathology"].astype("category")
    df["BIRADS"] = df["BIRADS"].astype("category")

    # Divide the dataset into two stratified samples
    target_variable = "BIRADS" if args.birads else "Pathology"

    for i in range(5):
        os.makedirs(os.path.join(args.output_dir, f"split{i}"), exist_ok=True)
        patients = (
            df.drop_duplicates("Case")
            .groupby([target_variable], group_keys=False)
            .apply(lambda x: x.sample(frac=0.5, random_state=i))["Case"]
            .to_list()
        )
        indices = df[df["Case"].isin(patients)].index
        df["Fold"] = 0
        df.loc[indices, "Fold"] = 1

        for fold in df["Fold"].unique():
            for class_name in df[target_variable].cat.categories:
                # Divide the images for each class into separate directories
                class_dir = os.path.join(
                    args.output_dir, f"split{i}", f"fold{fold}", str(class_name)
                )
                os.makedirs(class_dir, exist_ok=True)

                class_images = df.query(
                    f"{target_variable} == @class_name and Fold == @fold"
                )["ID"]
                for image in class_images:
                    shutil.copy(
                        os.path.join(args.input_dir, "Images", image + ".png"),
                        os.path.join(class_dir, image + ".png"),
                    )


if __name__ == "__main__":
    main()
