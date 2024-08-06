import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract():

    local_zip = r"C:\Users\karat\OneDrive\Documents\machine learning\datasets\air_pass_time_series.zip"
    extract_dir = r"C:\Users\karat\OneDrive\Documents\machine learning\datasets\air_pass_time_series"

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return os.path.join(extract_dir, "AirPassengers.csv")

def load_data(df,path):
    df = pd.read_csv(path)
    return df

def set_index(data):
    data['Month'] = pd.to_datetime(data['Month'])
    data.set_index('Month', inplace=True)

    return data

def rename(data, columns):
    data.rename(columns=columns, inplace=True)
    return data

def viz_data(data):
    colors = sns.color_palette()
    data.plot(style='-', figsize=(15,5), color=colors[0], title='Air Passengers')
    plt.show()

    return data