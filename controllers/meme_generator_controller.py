import os
import pandas as pd

def read_csv_file():
    #import dataset
    file_path = os.path.join(os.path.dirname(__file__), '../Datasets/csv_files/meme_captions_collection.csv')
    df = pd.read_csv(file_path)

    #specify no. of rows to be fetched
    df= df.head(3872)

    return df


def get_meme_titles(df: pd.DataFrame):
    #meme title using indexing
    meme_title= df.iloc[:,0:1]
    return meme_title

