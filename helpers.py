import os
import pandas as pd

def readfile(directory: str, file: str):
    df = pd.read_csv(os.getcwd() + '/' + directory + '/' + file)
    
    return df
