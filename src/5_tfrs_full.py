# %%
import json
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


def load_data(
    train_path: str, test_path: str, metadata_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    return train_data, test_data, metadata


# %%
def convert_to_tensorflow_df(data: pd.DataFrame) -> tf.data.Dataset:
    data_dict = {column: data[column].values for column in data.columns}
    return tf.data.Dataset.from_tensor_slices(data_dict)


# %%
class RatingPredictionModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()

        # user tower
        self.user_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="user_input")
        self.user_sl = tf.keras.layers.StringLookup(
            vocabulary=user_vocab, name="user_string_lookup"
        )(self.user_input)
        self.user_emb = tf.keras.layers.Embedding(
            len(user_vocab) + 1, 25, name="user_emb"
        )(self.user_sl)

        self.user_dense = tf.keras.layers.Dense(
            20, activation="relu", name="user_dense"
        )(self.user_emb)

        # movie tower
        self.movie_input = tf.keras.Input(
            shape=(1,), dtype=tf.string, name="movie_input"
        )
        self.movie_sl = tf.keras.layers.StringLookup(
            vocabulary=movie_vocab, name="movie_string_lookup"
        )(self.movie_input)
        self.movie_emb = tf.keras.layers.Embedding(
            len(movie_vocab) + 1, 25, name="movie_emb"
        )(self.movie_sl)

        self.movie_dense = tf.keras.layers.Dense(
            20, activation="relu", name="movie_dense"
        )(self.movie_emb)

        # Merging towers
        self.towers_multiplied = tf.keras.layers.Multiply(name="towers_multiplied")(
            [self.user_dense, self.movie_dense]
        )
        self.towers_dense = tf.keras.layers.Dense(
            10, activation="relu", name="towers_dense"
        )(self.towers_multiplied)
        self.output_node = tf.keras.layers.Dense(1, name="output_node")(
            self.towers_dense
        )

        # model definition
        self.model = tf.keras.Model(
            inputs={"user": self.user_input, "movie": self.movie_input},
            outputs=self.output_node,
        )

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features):
        return self.model(
            {
                "user": tf.strings.as_string(features["user_id"]),
                "movie": tf.strings.as_string(features["movie_id"]),
            }
        )

    def compute_loss(self, features, **kwargs):
        return self.task(labels=features["rating"], predictions=self(features))


if __name__ == "__main__":
    TRAIN_PATH = "local_data/train_data.csv"
    TEST_PATH = "local_data/test_data.csv"
    METADATA_PATH = "local_data/metadata.json"

    train_df, test_df, metadata = load_data(TRAIN_PATH, TEST_PATH, METADATA_PATH)

    user_vocab = [str(i) for i in metadata.get("user_id")]
    movie_vocab = [str(i) for i in metadata.get("movie_id")]

    train_df = convert_to_tensorflow_df(train_df)
    test_df = convert_to_tensorflow_df(test_df)

    # instantiate the model
    model = RatingPredictionModel()
    LEARNING_RATE= 2e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # train the model
    cached_train = train_df.shuffle(15000).batch(10000).cache()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", restore_best_weights=True, patience=5
    )

    model.fit(cached_train, epochs=10, callbacks=[early_stopping])

    # test the model
    cached_test = test_df.batch(5000).cache()
    model.evaluate(cached_test, return_dict=True)

    tf.keras.utils.plot_model(model.model, to_file="images/two_tower_simple.png")

    print(model.model.summary())

    model.save_weights("models/weights_simple")
