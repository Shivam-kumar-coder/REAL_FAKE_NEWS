import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
#from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

#st.title('News Detect 🔎')
#text=st.text_input("ENter your NEWS")
#but=st.button('Search')
df=pd.read_csv("Fake_Real_Data.csv")
df.drop_duplicates(inplace=True)

#v=TfidfVectorizer(stop_words='english',max_features=5000)
df['Label']=df['label'].map({'Fake': 0 ,'Real':1})

t=Tokenizer(num_words=5000)
t.fit_on_texts(df['Text'])
#seq=t.texts_to_sequences(df['Text'])

#import numpy as np

#x=pad_sequences(seq,maxlen=200)
#y=np.array(df['Label'])
#model.fit(x,y,epochs=5)
mode=load_model('fake_real_news.h5')
st.title('News Detect 🔎')
text=st.text_input("ENter your NEWS")
but=st.button('Search')

def classify(news):
    seq=t.texts_to_sequences([news])
    padded=pad_sequences(seq,maxlen=200)
    pred=mode.predict(padded)[0][0]
    return "REAL" if pred>0.5 else "FAKE"

if text:
    if but:
        #st.write("checking on running 🏃‍♀️")
        result=classify(text)
        st.write(result)
