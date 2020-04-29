import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from datetime import datetime


def convert(year,month,date):
    return int(datetime(year, month, date, 0, 0, 0).timestamp()*1000)

def convert_reverse(timestamp):
    dt_object = datetime.fromtimestamp(timestamp/1000)
    print("dt_object =", dt_object)
    return dt_object


def mapping_to_actual_data(dataframe_clustered,dataframe_actual):
    #dataframe_actual[label_to_transfer]=np.zeros((len(dataframe_actual)),dtype=int)
    list_df=[]
    for index,row in tqdm(dataframe_clustered.iterrows(),total=len(dataframe_clustered)):
        for ele in row['repeated messages']:
            #dataframe_actual.iloc[ele][label_to_transfer]=row[label_to_transfer]
            list_df.append((ele,row['preds'],row['pred_probab']))
            
    list_df.sort(key=lambda tup: tup[0])
    print(list_df[0:5])
    list_pred=[ele[1] for ele in list_df]
    list_probab=[ele[2] for ele in list_df]
    
    dataframe_actual['preds']=list_pred
    dataframe_actual['pred_probab']=list_probab
 
    return dataframe_actual


def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='macro'))
    avg.append(accuracy_score(y_true, y_pred, normalize=True))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support','accuracy']
    list_all=list(metrics_summary)
    list_all.append(cm.diagonal())
    class_report_df = pd.DataFrame(
        list_all,
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-2] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T
