# %%
import pandas as pd
from datetime import datetime
from uszipcode import SearchEngine

# %%
MOVIES_DATA_URL = "../data/movies.csv"
USERS_DATA_URL = "../data/users.csv"
RATINGS_DATA_URL = "../data/ratings.csv"

movies_df = pd.read_csv(MOVIES_DATA_URL, encoding="latin-1")
users_df = pd.read_csv(USERS_DATA_URL, encoding="latin-1")
ratings_df = pd.read_csv(RATINGS_DATA_URL, encoding="latin-1")
# %% movies.csv cleanup
print(movies_df.isna().any())
movies_df[["title", "movie_year", "genres"]] = movies_df.apply(
    lambda row: pd.Series(
        {
            "title": row["title"][:-7],
            "movie_year": int(row["title"][-5:-1]),
            "genres": (
                str(row["genres"]).split("|")
                if not pd.isnull(row["genres"])
                else list()
            ),
        }
    ),
    axis=1,
)

movies_df.rename(
    columns={"movie": "movie_id", "title": "movie_title", "genres": "movie_genre"},
    inplace=True,
)

print(movies_df)

# %%
# users.csv clean up
# change gender to 1/0 (F/M)
# we need to standardize the zipcode column
# also take extra information from zipcode like city and state

users_df.columns = "user_id user_gender user_age user_job user_zip".split()
zip_search = SearchEngine()


def user_features(row):
    zip = int(str(row["user_zip"])[:5]) if not pd.isnull(row["user_zip"]) else 0
    zip_search_dict = zip_search.by_zipcode(zip).to_dict()
    return pd.Series(
        {
            "user_gender": int(row["user_gender"] == "F"),
            "user_city": zip_search_dict.get("major_city", ""),
            "user_state": zip_search_dict.get("state", ""),
            "user_zipcode": zip,
        }
    )


users_df["user_gender user_city user_state user_zip".split()] = users_df.apply(
    user_features, axis=1
)

print("User df after cleaning:\n")
print(users_df)

# %%
# ratings.csv clean up
# convert timestamp from epoch to date columns + time
# %%
ratings_df.columns = "user_id movie_id rating timestamp".split()
timestamp = pd.to_datetime(ratings_df["timestamp"], unit="s")
ratings_df["ratings_month"] = timestamp.dt.month
ratings_df["ratings_day"] = timestamp.dt.dayofweek
ratings_df["ratings_hour"] = timestamp.dt.hour

print(ratings_df)

# %%
# join the tables and handle missing values
full_df = ratings_df.join(
    users_df.set_index("user_id"), on="user_id", how="inner"
).join(movies_df.set_index("movie_id"), on="movie_id", how="inner")

print(f"Total rows in full df: {len(full_df.index)}")
print(full_df.head())

# %%
# check for missing values
print(pd.isna(full_df).any())

# we have NAs in user_state and user_city.
MISSING_STRING = "XX"
full_df["user_city user_state".split()] = full_df[
    "user_city user_state".split()
].fillna(MISSING_STRING)
print(pd.isna(full_df).any())

# %%
full_df.to_parquet("../local_data/full_data.parquet")
