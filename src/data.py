import pandas as pd

from .config import DATA_PATH, LEAKAGE_COLUMNS, TARGET_COLUMN


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the weather dataset using the repository default path."""
    return pd.read_csv(path, na_values=["NA"])


def prepare_dataset(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target, excluding columns that leak tomorrow's rain."""
    data = raw_data.copy()
    data = data.dropna(subset=[TARGET_COLUMN])

    y = data[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    forbidden = [TARGET_COLUMN, *LEAKAGE_COLUMNS]
    x = data.drop(columns=[col for col in forbidden if col in data.columns])

    if "Date" in x.columns:
        date = pd.to_datetime(x["Date"], errors="coerce")
        x["Year"] = date.dt.year
        x["Month"] = date.dt.month
        x["Day"] = date.dt.day
        x = x.drop(columns=["Date"])

    return x, y
