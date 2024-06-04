# TODO: design categoricla features vocab
# TODO: preprocess genres and titles

# %% Imports
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer

# %% Data loading

data = pd.read_parquet("../local_data/full_data.parquet")
data.head()

# %% Test train split
data = data.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
data["is_train"] = data.index % 20 == 0

print(f"Test split has {len(data[data['is_train'] == 1])} rows")

# %% Preprocessing genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data.movie_genre)
genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
genres_encoded_df

#! Possible issue here with mismatched colums. Come back here if issues.
genre_columns = genres_encoded_df.columns.values


def clean_genres(genre):
    genre = re.sub(r"[\W_]", "", genre)
    genre = genre.lower()
    return f"genre_{genre}"


genre_columns_clean = [clean_genres(genre) for genre in genre_columns]
genres_encoded_df.columns = genre_columns_clean

data = pd.concat([data, genres_encoded_df], axis=1)
data

# %%
