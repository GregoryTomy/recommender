# TODO: design categoricla features vocab
# TODO: preprocess genres and titles

# %% Imports
from typing import List
import gensim
import pandas as pd
import numpy as np
import gensim.downloader
import json
import re
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.parsing.preprocessing import *

# %% Data loading

data = pd.read_parquet("../local_data/full_data.parquet")
data.head()

data = data.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
data["is_test"] = data.index % 20 == 0

print(f"Test split has {len(data[data['is_test'] == 1])} rows")

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

# %% Processing titles
word2vec = gensim.downloader.load("glove-twitter-25")


# %%
def safe_word_embed(word: string, length=25) -> np.array:
    """Function to safely embed a word as the model requires words to be
    lower cased.
    Args:
        word (string): title string to embed
        length (int, optional): Defaults to 25.

    Returns:
        np.array: the embedding vector
    """
    try:
        return word2vec[word.lower()]
    except:
        return np.zeros(length)


def gensim_embed_processing(title: string) -> List[str]:
    title = strip_non_alphanum(title)
    title = split_alphanum(title)
    title = remove_stopwords(title)
    title = strip_punctuation(title)
    title = strip_multiple_whitespaces(title)

    return title.split(" ")


def embed_title(title: string) -> pd.Series:
    words = gensim_embed_processing(title)
    word_embeddings = np.array([safe_word_embed(word) for word in words])
    title_embeddings = word_embeddings.sum(axis=0)

    return pd.Series(
        {
            f"title_emb_{idx}": embedding
            for idx, embedding in enumerate(title_embeddings)
        }
    )


embeddings_df = data.movie_title.apply(embed_title)
data = pd.concat([data, embeddings_df], axis=1)
print(data.columns)

# %% Test tran split
data = data.drop(columns=["user_zip", "timestamp", "movie_title"])
test_data = data[data.is_test].drop(columns="is_test")
train_data = data[~data.is_test].drop(columns="is_test")

print(test_data.shape)
print(train_data.shape)


test_data.to_csv("../local_data/test_data.csv", index=False)
train_data.to_csv("../local_data/train_data.csv", index=False)

# %% Metadata for the model. See eda.ipynb long-tail distribution section
metadata = dict()
metadata["title_embedding_size"] = 25  # size of embedding of glove-twitter-25
metadata["na_string"] = "XX"
metadata["genres"] = genre_columns_clean
metadata["ages"] = data.user_age.unique().tolist()
metadata["occupations"] = data.user_job.unique().tolist()

metadata["user_id"] = data.user_id.value_counts().index.to_list()[:2500]
metadata["movie_id"] = data.movie_id.value_counts().index.to_list()[:1200]
metadata["user_city"] = (
    data.user_city[~(data.user_city == "XX")].value_counts().index.to_list()[:500]
)
metadata["user_state"] = (
    data.user_state[~(data.user_state == "XX")].value_counts().index.to_list()[:20]
)

# %% save the metadata to json file

with open("../local_data/metadata.json", "w+") as file:
    json.dump(metadata, file)
# %%
