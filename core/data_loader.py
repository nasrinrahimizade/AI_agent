import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Reads a CSV file from `path` into a pandas DataFrame.
    """
    df = pd.read_csv(path)
    # you can do any cleaning here (e.g. df.dropna(), parse dates, etc.)
    return df
