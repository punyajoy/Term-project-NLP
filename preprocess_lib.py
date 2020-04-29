import pandas as pd
import cltk
import random
from cltk.tokenize.sentence import TokenizeSentence
from cltk.tokenize.word import WordTokenizer
from nltk.corpus import stopwords 
import demoji
import re
import string
import emoji
from tqdm import tqdm_notebook

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize

#demoji.download_codes()


##### support for hindi tokenizations present along with english. Target has ben to keep the same level of preprocessing to both the text.


def re_sub(pattern, repl,text):
    return re.sub(pattern, repl, text)



stop_words_hin=["अंदर","अत","अदि","अप","अपना","अपनि","अपनी","अपने","अभि","अभी","आदि","आप","इंहिं","इंहें","इंहों","इतयादि","इत्यादि","इन","इनका","इन्हीं","इन्हें","इन्हों","इस","इसका","इसकि","इसकी","इसके","इसमें","इसि","इसी","इसे","उंहिं","उंहें","उंहों","उन","उनका","उनकि","उनकी","उनके","उनको","उन्हीं","उन्हें","उन्हों","उस","उसके","उसि","उसी","उसे","एक","एवं","एस","एसे","ऐसे","ओर","और","कइ","कई","कर","करता","करते","करना","करने","करें","कहते","कहा","का","काफि","काफ़ी","कि","किंहें","किंहों","कितना","किन्हें","किन्हों","किया","किर","किस","किसि","किसी","किसे","की","कुछ","कुल","के","को","कोइ","कोई","कोन","कोनसा","कौन","कौनसा","गया","घर","जब","जहाँ","जहां","जा","जिंहें","जिंहों","जितना","जिधर","जिन","जिन्हें","जिन्हों","जिस","जिसे","जीधर","जेसा","जेसे","जैसा","जैसे","जो","तक","तब","तरह","तिंहें","तिंहों","तिन","तिन्हें","तिन्हों","तिस","तिसे","तो","था","थि","थी","थे","दबारा","दवारा","दिया","दुसरा","दुसरे","दूसरे","दो","द्वारा","न","नहिं","नहीं","ना","निचे","निहायत","नीचे","ने","पर","पहले","पुरा","पूरा","पे","फिर","बनि","बनी","बहि","बही","बहुत","बाद","बाला","बिलकुल","भि","भितर","भी","भीतर","मगर","मानो","मे","में","यदि","यह","यहाँ","यहां","यहि","यही","या","यिह","ये","रखें","रवासा","रहा","रहे","ऱ्वासा","लिए","लिये","लेकिन","व","वगेरह","वरग","वर्ग","वह","वहाँ","वहां","वहिं","वहीं","वाले","वुह","वे","वग़ैरह","संग","सकता","सकते","सबसे","सभि","सभी","साथ","साबुत","साभ","सारा","से","सो","हि","ही","हुअ","हुआ","हुइ","हुई","हुए","हे","हें","है","हैं","हो","होता","होति","होती","होते","होना","होने","हैं।","नही","वो","क्या"]



stop_words_eng=stopwords.words('english')   ### all english stop words
punct_list=list(string.punctuation)+["--","..","...","''","``","'s"]  ## punctuations that needs to be removed 
stop_all=stop_words_eng+stop_words_hin+punct_list


def english_preprocess(text,multiple_sentences=True,stop_word_remove=True, tokenize_word=True, tokenize_sentence=True):
    tknzr = TweetTokenizer()
    text=text.lower()
    if(multiple_sentences):
        english_sentences=sent_tokenize(text)
    else:
        english_sentences=[text]
    all_sentences=[] 
    for sent in english_sentences:
        #print(sent)
        sent = re.sub(r"http\S+", "", sent)   
        #print(sent)
        sent = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "",sent)
        sent = emoji.demojize(sent)
        sent = re_sub(r"[:\*]", " ",sent)
        
        
        if(tokenize_word):
            if(stop_word_remove):
                words=[]
                for word in tknzr.tokenize(sent):
                        if(word not in stop_all):
                            words.append(word)
            else:
                words = tknzr.tokenize(sent)
            all_sentences.append(words)
      
        
        else:
            if(stop_word_remove):
                words=[]
                for word in tknzr.tokenize(sent):
                        if(word not in stop_all):
                            words.append(word)
                sent=" ".join(words)
            all_sentences.append(sent)
    if(tokenize_sentence):
        return all_sentences
    elif(tokenize_word):
        para=[]
        for sent in all_sentences:
            para+=sent
        return para
    else:
        return " . ".join(all_sentences)
        

        
def preprocess_multi(text,lang,multiple_sentences=True,stop_word_remove=True, tokenize_word=True, tokenize_sentence=True):
    
    tokenizer = TokenizeSentence('hindi')
    word_tokenizer = WordTokenizer('multilingual')

    if(multiple_sentences):
        hindi_sentences=tokenizer.tokenize_sentences(text)
    else:
        hindi_sentences=[text]
    all_sentences=[] 
    for sent in hindi_sentences:
        #print(sent)
        sent = re.sub(r"http\S+", "", sent)   
        #print(sent)
        sent = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "",sent)
#         sent = emoji.demojize(sent)
#         sent = re_sub(r"[:\*]", " ",sent)
        
        
        if(tokenize_word):
            if(stop_word_remove):
                words=[]
                for word in word_tokenizer.tokenize(sent):
                        if(word not in stop_all):
                            words.append(word)
            else:
                words = word_tokenizer.tokenize(sent)
            all_sentences.append(words)
      
        
        else:
            if(stop_word_remove):
                words=[]
                for word in word_tokenizer.tokenize(sent):
                        if(word not in stop_all):
                            words.append(word)
                sent=" ".join(words)
            all_sentences.append(sent)
    if(tokenize_sentence):
        return all_sentences
    elif(tokenize_word):
        para=[]
        for sent in all_sentences:
            para+=sent
        return para
    else:
        if(lang=='hi'):
           return " | ".join(all_sentences)
        if(lang=='en'):
           return " . ".join(all_sentences)
        

            
        





