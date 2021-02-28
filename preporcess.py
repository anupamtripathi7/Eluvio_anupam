import pandas as pd
import os


data_path = 'data'


def imdb_one_hot(file_apth):
    df = pd.read_csv(file_apth, sep='\t')
    df = df.join(df.genres.str.get_dummies(sep=','))
    df.to_csv(os.path.join(data_path, 'title.basics.tsv'), sep='\t')


if __name__ == '__main__':
    imdb_one_hot(os.path.join(data_path, 'title.basics.tsv'))