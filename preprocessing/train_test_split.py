"""Perform train-test split on data, stratified by label"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Random seed for reproducibility
random_state = 12345

test_size = 0.2
data_dir = Path("../data")

# Dataframe with .wav filenames and labels
file_directory_df = pd.read_csv(data_dir / "dataset_file_directory.csv")
pd.testing.assert_index_equal(
    file_directory_df.index, pd.RangeIndex(len(file_directory_df))
)

train_df, test_df, train_idx, test_idx = train_test_split(
    file_directory_df,
    file_directory_df.index,
    stratify=file_directory_df.Label,
    test_size=test_size,
    shuffle=True,
    random_state=random_state,
)
pd.testing.assert_frame_equal(train_df, file_directory_df.iloc[train_idx])
pd.testing.assert_frame_equal(test_df, file_directory_df.iloc[test_idx])

# Add a column is_test, which is 1 if a row is in the test set
file_directory_df["is_test"] = 0
file_directory_df.loc[test_idx, "is_test"] = 1
assert np.isclose(file_directory_df.is_test.mean(), 0.2, rtol=0.001)

output_file = data_dir / "directory_w_train_test.csv"
file_directory_df.to_csv(output_file, index=False)

print("Train-test split completed, saved to", output_file)
