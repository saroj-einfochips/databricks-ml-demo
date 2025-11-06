import pandas as pd

def save_df_csv(df, path):
    df.to_csv(path, index=False)