from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocess_lib import *
from empath import Empath
lexicon = Empath()
from tqdm import tqdm
analyzer = SentimentIntensityAnalyzer()
from nltk.tokenize import sent_tokenize

def sentiment_text_paragraph(text):
    if(type(text)!=str):
        return 0
    sentences=preprocess_multi(text,'en',multiple_sentences=True,stop_word_remove=True, tokenize_word=False, tokenize_sentence=True)
    sentences=sent_tokenize(text)
    sum1=0
    for sent in sentences:
        vs = analyzer.polarity_scores(sent)
        sum1+=vs["compound"]
    if(len(sentences)>0):
        return sum1/len(sentences)
    else:
        return 0



def generate_word_cloud(wp_sort,name,end_date=0,post_event=False,number_of_days=5): 
    if(post_event):
        days_after=end_date+(days*24*60*60*1000)
        wp_sort=wp_sort[wp_sort['timestamp'].between(end_date,days_after, inclusive=True)]
    list1=" "
    for element in list(wp_sort["translated"].values):
        try:
            list1+=element+" "
        except:
            pass
    stop_words = set(STOPWORDS)
    stop_words.update(["Surf","Excel","reply","message", "Add","group","year","old", "Narendra","Modi" "Prime","Minister","detailed","news","books","thumbsup","type","BJP","Rahul","Gandhi","Hindu","Muslim","Modi","ji"])
    plt.figure( figsize=(100,300), facecolor='k')
    wordcloud = WordCloud(stopwords=stop_words,width=1600, height=800,background_color="white",colormap='magma').generate(list1)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if(post_event):
        plt.savefig('Results/'+'post_event_'+name+'.png',dpi=400)
    else:
        plt.savefig('Results/'+name+'.png',dpi=400)

def action_of_group(df,action_list):
    count=0
    for index,row in df.iterrows():
        if(row["action"] in action_list):
            count+=1
    return count


def count_symbols(df,symbols):
    count=0
    for index,row in df.iterrows():
        for sym in symbols:
            if sym in row["message_text"]:
                count+=1
                break
    return count


def avg_length(df):
    sum1=0
    for index,row in df.iterrows():
            sum1+=len(row["message_text"])
    
    return sum1/len(df)

from features import *

def sentiment_group(df):
    sentiment=0
    for index,row in df.iterrows():
        sentiment+=sentiment_text_paragraph(row["translated"])
    if(len(df)>0):
        return sentiment/len(df)
    else:
        return 0



def get_empath_categories(df,normalize=True): 
    list1=" "
    for element in tqdm(list(df["translated"].values)):
        try:
            list1+=element+" "
        except:
            pass
    dict_lexicon=lexicon.analyze(list1, normalize=normalize)
    return dict_lexicon

        



