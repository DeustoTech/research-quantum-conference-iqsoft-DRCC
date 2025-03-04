import pandas as pd


def load_dataset(filename):
    df = pd.read_csv(filename)
    return df
