import pandas as pd
import pathlib as pl
import csv

def main():
    # normalize the preprocessed data except for the A3-data (we will use the original data for that)      
    for txt_file in pl.Path("datasets/preprocessed").glob('*.csv'):
        if(txt_file.name.startswith("A3-data")):
            continue
        df = pd.read_csv(f"datasets/preprocessed/{txt_file.name}", encoding='utf8', sep=';')
        for col in df.columns:
            if((df[col].max() - df[col].min()) != 0):
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                df[col] = 0.0
        df.to_csv(f"datasets/normalized/{txt_file.name}", encoding='utf8', sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == '__main__':
    main()
