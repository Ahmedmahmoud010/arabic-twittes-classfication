#!/usr/bin/env python
# coding: utf-8

# In[136]:


#liberaries for reading file and make df 
import os
import pandas as pd
import re
#liberaries for text cleaning 
import codecs
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[68]:


#read negative tweets 
RootDir=os.chdir("/Users/ahmed/Documents/Twitter/Negative")
file_names_negative = os.listdir(RootDir)


# In[47]:


#read positive tweets 
RootDir=os.chdir("/Users/ahmed/Documents/Twitter/Positive")
file_names_positive = os.listdir(RootDir)


# In[48]:


file_names_negative


# In[49]:


len(file_names_positive),len(file_names_negative)


# In[50]:


all_reviews=file_names_positive+file_names_negative


# In[51]:


len(all_reviews)


# In[52]:


#make one data frame for all positive files 
positive_df=pd.DataFrame([],columns=['text'])
for i in file_names_positive:
    data = pd.read_csv(i,names=['text']) 
    positive_df=pd.concat([positive_df,data],ignore_index=True)


# In[99]:


#add 1 to represent positive opinin in y axis 
positive_df['reviews']=1


# In[98]:


#make one data frame for all negative files
negative_df=pd.DataFrame([],columns=['text'])
for i in file_names_negative:
    try:
        data = pd.read_csv(i,names=['text']) 
        negative_df=pd.concat([negative_df,data],ignore_index=True)
    except:
        pass        


# In[101]:


#add -1 to represent negative opinin in y axis
negative_df['reviews']=-1


# In[106]:


#cobiend df(positive and negative in one df)
total_reviews=negative_df.append(positive_df,ignore_index=True)


# In[107]:


total_reviews


# In[133]:


#function to remove any char expect 28 letter then remove stoping words then return to source word (steaming) 
stopwords = codecs.open('/Users/ahmed/Desktop/arabic_list.txt','r',encoding='utf-8').read().split('\n')
isri = ISRIStemmer()
def clean(text):
    global stopwords
    global isri
    text = re.sub(r'[^ء-ي]+',' ',text)
    clean_words = []
    for word in text.split():
        if word not in stopwords:
            word = isri.stem(word)
            clean_words.append(word)
    text = ' '.join(clean_words)
    return text


# In[134]:


#apply clean function to the text 
total_reviews['denoise']=total_reviews['text'].apply(clean)


# In[135]:


total_reviews


# In[161]:


#make count vectroise array
cv = CountVectorizer()
cv.fit(total_reviews['denoise'])
X = cv.transform(total_reviews['denoise']).toarray()


# In[170]:


#liberires for split and logistic regresion 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#liberires for tfidf vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[167]:


#split the data 
x_train,x_test,y_train,y_test = train_test_split(X,total_reviews['reviews'],test_size=0.25)


# In[171]:


def predict_text(text,algorithm):
    if algorithm=='tfidf':
    #modling using vectorizer 
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(total_reviews['denoise']).toarray()
        lr = LogisticRegression()
        lr.fit(x_train,y_train)
        lr.score(x_test,y_test)
    elif  algorithm=='logistic': 
        #modling using logistic regression 
        lr = LogisticRegression()
        lr.fit(x_train,y_train)
        lr.score(x_test,y_test)
    text = clean(text)
    x_t  = cv.transform([text]).toarray()
    return lr.predict_proba(x_t)


# In[173]:


predict_text(text='اقذر انسان',algorithm='tfidf')


# In[147]:




