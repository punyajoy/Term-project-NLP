import pandas as pd
import numpy as np
from tqdm import tqdm

def mapping_to_actual_data(dataframe_clustered,dataframe_actual,label_to_transfer):
    #dataframe_actual[label_to_transfer]=np.zeros((len(dataframe_actual)),dtype=int)
    list_df=[]
    for index,row in tqdm(dataframe_clustered.iterrows(),total=len(dataframe_clustered)):
        for ele in row['repeated messages']:
            #dataframe_actual.iloc[ele][label_to_transfer]=row[label_to_transfer]
            list_df.append((ele,row[label_to_transfer]))
            
    list_df.sort(key=lambda tup: tup[0])
    print(list_df[0:5])
    list_temp=[ele[1] for ele in list_df]
    dataframe_actual[label_to_transfer]=list_temp
    return dataframe_actual