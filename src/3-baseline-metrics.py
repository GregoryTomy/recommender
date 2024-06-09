import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def rmse(actual: pd.Series, predicted: np.array) -> float:
    errors = actual - predicted
    squared_errors = np.square(errors)

    return np.mean(squared_errors)


def random_guessing(data: pd.Series) -> float:
    random_predictions = np.random.randint(1, 5 + 1, size=len(data))
    return rmse(data, random_predictions)


def weighted_sampling(data: pd.Series) -> float:
    weights = data.value_counts(normalize=True)
    sampled_predictions = np.random.choice(
        weights.index, size=len(data), p=weights.values
    )
    return rmse(data, sampled_predictions)


def majority_class(data: pd.Series) -> float:
    mode = data.mode()[0]
    majority_predictions = np.full(len(data), mode)
    return rmse(data, majority_predictions)


def mean_value(data: pd.Series) -> float:
    mean = data.mean()
    mean_predictions = np.full(len(data), mean)
    return rmse(data, mean_predictions)


if __name__ == "__main__":
    TRAIN_DATA_PATH = "local_data/train_data.csv"
    train_ratings = pd.read_csv(TRAIN_DATA_PATH).rating

    logging.info("Calculating RMSE for different baseline methods...")

    random_rmse = random_guessing(train_ratings)
    logging.info(f"Random guessing RMSE:{random_rmse}")

    weighted_rmse = weighted_sampling(train_ratings)
    logging.info(f"Weighted sampling RMSE:{weighted_rmse}")

    majority_rmse = majority_class(train_ratings)
    logging.info(f"Majority class RMSE:{majority_rmse}")

    mean_rmse = mean_value(train_ratings)
    logging.info(f"Mean RMSE:{mean_rmse}")

    baseline_df = pd.DataFrame(
        {
            "Method": "Random-guessing Weighted-sampling Majority-class Mean-value".split(),
            "RMSE": [random_rmse, weighted_rmse, majority_rmse, mean_rmse],
        }
    )

    baseline_df.to_csv("local_data/baseline_rmse.csv", index=False)
    logging.info("Baseline RMSEs saved.")
