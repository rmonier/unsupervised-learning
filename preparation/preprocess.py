import pandas as pd
import pathlib as pl
import csv

def main():

    # A3 DATASETS (no processing, only convertion to csv and placement in the right folder)

    txt_file = pl.Path("A3-data.txt")
    df = pd.read_csv(f"datasets/raw/{txt_file.name}", encoding='utf8', sep=',').astype(float)
    df.to_csv(f"datasets/preprocessed/{txt_file.stem}.csv", encoding='utf8', sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)

    # OUR DATASET

    df = pd.read_csv("datasets/raw/A3-top10s.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    del df[df.columns[0]]
    df.drop(['pop', 'title', 'artist', 'year'], axis=1, inplace=True)

    # Merge some top genres to reduce the number of classes
    df.loc[df['top genre'] == 'dance pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'canadian pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'acoustic pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'acoustic pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'art pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'atl hip hop', 'top genre'] = 'hip hop'
    df.loc[df['top genre'] == 'australian pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'australian hip hop', 'top genre'] = 'hip hop'
    df.loc[df['top genre'] == 'australian dance', 'top genre'] = 'dance'
    df.loc[df['top genre'] == 'barbadian pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'baroque pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'belgian edm', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'big room', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'big room', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'alternative r&b', 'top genre'] = 'r&b'
    df.loc[df['top genre'] == 'canadian contemporary r&b', 'top genre'] = 'r&b'
    df.loc[df['top genre'] == 'canadian hip hop', 'top genre'] = 'hip hop'
    df.loc[df['top genre'] == 'canadian latin', 'top genre'] = 'latin'
    df.loc[df['top genre'] == 'candy pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'chicago rap', 'top genre'] = 'rap'
    df.loc[df['top genre'] == 'colombian pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'celtic rock', 'top genre'] = 'rock'
    df.loc[df['top genre'] == 'contemporary country', 'top genre'] = 'country'
    df.loc[df['top genre'] == 'complextro', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'country road', 'top genre'] = 'country'
    df.loc[df['top genre'] == 'detroit hip hop', 'top genre'] = 'hip hop'
    df.loc[df['top genre'] == 'downtempo', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'electro', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'electro house', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'electronic trap', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'electropop', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'escape room', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'folk-pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'french indie pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'hollywood', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'house', 'top genre'] = 'edm'
    df.loc[df['top genre'] == 'indie pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'irish singer-songwriter', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'metropopolis', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'moroccan pop', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'neo mellow', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'permanent wave', 'top genre'] = 'pop'
    df.loc[df['top genre'] == 'tropical house', 'top genre'] = 'edm'

    # We want to predict if its 'edm' or not
    df.loc[df['top genre'] == 'edm', 'top genre'] = 1
    df.loc[df['top genre'] != 1, 'top genre'] = 0

    df["top genre"] = pd.factorize(df["top genre"])[0]  # convert class labels to integers
    df = df.astype(float)
    df = df[[c for c in df if c not in ["top genre"]] + ["top genre"]]  # move top genre to the end

    # filter out the outliers
    for i in range(len(df.columns)):
        data_mean, data_std = df.iloc[:,i].mean(), df.iloc[:,i].std()
        # with try and error -> 5 seems to be a good value
        cut_off = data_std * 5
        lower, upper = data_mean - cut_off, data_mean + cut_off
        df=df[(df.iloc[:,i] >= lower) & (df.iloc[:,i] <= upper)]
    
    df.to_csv("datasets/preprocessed/A3-top10s.csv", encoding='utf8', sep=';', index=False, quoting=csv.QUOTE_NONNUMERIC)

if __name__ == '__main__':
    main()
