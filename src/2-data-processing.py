# %% Imports
import gensim
import pandas as pd
import numpy as np
import gensim.downloader
import json
import re
import logging

from sklearn.preprocessing import MultiLabelBinarizer
from gensim.parsing.preprocessing import *
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# %% Data loading
def load_data(file_path: str) -> pd.DataFrame:
    """_summary_

    Args:
        file_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logging.info("Loading data from parquet file.")
    data = pd.read_parquet(file_path)
    data = data.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    data["is_test"] = data.index % 20 == 0
    logging.info(f"Test split has {len(data[data['is_test'] == 1])} rows")
    return data


def clean_genres(genre):
    genre = re.sub(r"[\W_]", "", genre)
    genre = genre.lower()
    return f"genre_{genre}"


def preprocess_genres(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preprocessing movie genres.")
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data.movie_genre)
    genres_encoded_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    #! Possible issue here with mismatched colums. Come back here if issues.
    genre_columns = genres_encoded_df.columns.values

    genre_columns_clean = [clean_genres(genre) for genre in genre_columns]
    assert all(
        column.startswith("genre_") for column in genre_columns_clean
    ), "Genre columns not correctly encoded."

    genres_encoded_df.columns = genre_columns_clean

    data = pd.concat([data, genres_encoded_df], axis=1)

    return data


def load_word2vec_model() -> gensim.models.KeyedVectors:
    logging.info("Loading GloVe model")
    word2vec = gensim.downloader.load("glove-twitter-25")
    return word2vec


def safe_word_embed(
    word: string, model: gensim.models.KeyedVectors, length=25
) -> np.array:
    """Function to safely embed a word as the model requires words to be
    lower cased.
    Args:
        word (string): title string to embed
        length (int, optional): Defaults to 25.

    Returns:
        np.array: the embedding vector
    """
    try:
        return model[word.lower()]
    except:
        logging.warning(f"Word {word} not found in GloVe model. Returning zeros.")
        return np.zeros(length)


def gensim_embed_processing(title: string) -> List[str]:
    title = strip_non_alphanum(title)
    title = split_alphanum(title)
    title = remove_stopwords(title)
    title = strip_punctuation(title)
    title = strip_multiple_whitespaces(title)

    return title.split(" ")


def embed_title(title: string, model: gensim.models.KeyedVectors) -> pd.Series:
    words = gensim_embed_processing(title)
    word_embeddings = np.array([safe_word_embed(word, model) for word in words])
    title_embeddings = word_embeddings.sum(axis=0)

    return pd.Series(
        {
            f"title_emb_{idx}": embedding
            for idx, embedding in enumerate(title_embeddings)
        }
    )


def process_titles(
    data: pd.DataFrame, model: gensim.models.KeyedVectors
) -> pd.DataFrame:
    logging.info("Embedding movie titles.")
    embeddings_df = data.movie_title.apply(embed_title, model=model)

    return pd.concat([data, embeddings_df], axis=1)


def train_test_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Splitting data into test and train sets.")
    data = data.drop(columns=["user_zip", "timestamp", "movie_title"])
    test_data = data[data.is_test].drop(columns="is_test")
    train_data = data[~data.is_test].drop(columns="is_test")

    return train_data, test_data


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    test_data.to_csv("local_data/test_data.csv", index=False)
    train_data.to_csv("local_data/train_data.csv", index=False)
    logging.info("Training and testing data saved to CSV files.")


# %% Metadata for the model. See eda.ipynb long-tail distribution section
def create_metadata(data: pd.DataFrame, title_embedding_size: int = 25) -> Dict:
    logging.info("Creating metadata for the model.")
    metadata = dict()
    metadata["title_embedding_size"] = (
        title_embedding_size  # size of embedding of glove-twitter-25
    )
    metadata["na_string"] = "XX"
    metadata["genres"] = data.columns[data.columns.str.startswith("genre_")].to_list()
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
    logging.info("Metadata creation completed.")
    return metadata


def save_metadata(metadata: Dict, file_path: str) -> None:
    logging.info("Saving metadata to JSON file.")
    with open(file_path, "w+") as file:
        json.dump(metadata, file)
    logging.info("Saved metadata to JSON file.")


if __name__ == "__main__":
    DATA_PATH = "local_data/full_data.parquet"
    METADATA_PATH = "local_data/metadata.json"

    data = load_data(DATA_PATH)
    assert not data.empty, "Data loading failed, dataframe is empty"

    data = preprocess_genres(data)

    word2vec_model = load_word2vec_model()

    data = process_titles(data, word2vec_model)

    train_data, test_data = train_test_split(data)
    assert (
        not train_data.empty and not test_data.empty
    ), "Data split failed, one of the datasets is empty"

    logging.info(f"Training data shape: {train_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}")

    save_data(train_data, test_data)

    metadata = create_metadata(data)
    save_metadata(metadata, METADATA_PATH)
